#!/usr/bin/env python
"""
This file is used to perform simulations. It includes functions to generate movie, reconstruct 
traces and compute metrics.
@author: @caichangjia
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from utils import compute_spnr, match_spikes_greedy, compute_F1
from utils import load_movie, mov_interpolation, generate_streak_mov, generate_streak_mov_masks, hals, signal_filter, normalize
from utils import denoise_spikes

mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False, 
                     'xtick.major.size': 7, 
                     'ytick.major.size': 7})
#%matplotlib qt

#%% load movie
#mov = load_movie('/home/nel/CODE/temporal_optical_encoder/voltage_movie/raw/403106_3min_1.tiff')
data_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Datasets/changjia_compressive_microscope/voltage_movie'
save_folder = '/home/nel/CODE/compressive_micro/simulation/result'
mov = load_movie(data_folder +'/403106_3min_1_motion_corrected.tiff')
mov_raw = mov.copy()
mask = load_movie(data_folder +'/403106_3min_1_motion_corrected_mrcnn_ROIs.hdf5')

#%% compute groundtruth traces and remove neurons
init_frame = 2000

com = []
for mm in mask:
    com.append([np.where(mm>0)[0].mean(), np.where(mm>0)[1].mean()])
for i in range(len(mask)):
    for j in range(len(mask)):
        if i < j:
            yy = np.abs(com[i][0] - com[j][0])
            xx = np.abs(com[i][1] - com[j][1])
            if (xx < 4) and (yy < 30):
                print((i, j))
#del_overlap = np.array([5, 9, 12, 13, 21])
del_overlap = np.array([8, 9, 12, 14, 22])
mask = np.delete(mask, del_overlap, axis=0)

C_gt = (mov_raw[init_frame:].reshape((mov_raw[init_frame:].shape[0], -1))) @ (mask.reshape((mask.shape[0], -1)).T)
C_gt = C_gt.T
C_gt = np.array([-signal_filter(normalize(c), freq=1/3, fr=400) for c in C_gt])
C_gt = normalize(C_gt)

#%%
plt.imshow(mask.sum(0))
com = []
for mm in mask:
    com.append([np.where(mm>0)[0].mean(), np.where(mm>0)[1].mean()])
for idx, com1 in enumerate(com):
    plt.text(int(com1[1]), int(com1[0]), f'{idx}', c='red')

#%% functions for generate streak movie, reconstruct and compute metrics
def generate_streak_movie(mov, mask, init_frame, cr=10, size=5, 
                          poisson=False):
    print(f'compression ratio: {cr}')
    print(f'size: {size}')
    nx = mov.shape[1]
    ny = mov.shape[2] 
    nn = mask.shape[0]

    # initialize spatial footprints
    print('initialize spatial footprint')
    X = mov[:init_frame].copy().transpose([1, 2, 0])
    A = mask.copy()
    A = A / (np.linalg.norm(A, axis=(1, 2), ord='fro')[:, None, None])
    A = A.transpose([1, 2, 0])
    A = A.reshape((-1, A.shape[-1]), order='F')
    A = A.astype('float64')
    T = X.shape[-1]; n_comp = A.shape[-1]
    C = np.ones((n_comp, T))
    C = C / C.sum(1)[:, None]    
    A_init, C = hals(Y=X, A=A, C=C, b=None, f=None, bSiz=None, maxIter=5, update_shape=True)
    A_init = A_init.reshape((nx, ny, n_comp), order='F')
    
    # interpolate the movie  
    print('interpolate the movie')
    mov_inter = mov_interpolation(mov=mov[init_frame:], factor=5, method='piecewise_linear')
    mov = mov_inter
    
    # DMD patterned illumination
    print('DMD pattern illumination')
    T_raw = mov.shape[0]
    mov = mov.reshape((T_raw, -1), order='F')
    temp = mask.reshape((mask.shape[0], -1), order='F').sum(axis=0)
    mov[:, temp==0] = 0
    mov = mov.reshape((T_raw, nx, ny), order='F')
    
    # generate streak movie with compressed ratio
    print('generate streak movie')
    mov_temp, m_img = generate_streak_mov(mov, cr=cr, size=size)
    if poisson:
        mov_streak = np.random.poisson(mov_temp)
    else:
        mov_streak = mov_temp
        
    # generate masks
    print('generate streak masks')
    A = generate_streak_mov_masks(A_init, cr=cr, size=size)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(mov_raw.mean(0), cmap='gray')
    plt.axis('off')
    plt.title('Raw movie')

    plt.subplot(2, 3, 2)
    plt.imshow(mask.sum(0), cmap='gray')
    plt.axis('off')
    plt.title('DMD masks')

    plt.subplot(2, 3, 3)
    plt.imshow(A_init.sum(2), cmap='gray')
    plt.axis('off')
    plt.title('Neuron spatial masks')
    
    plt.subplot(2, 3, 4) 
    plt.imshow(mov[:3000].mean(0), cmap='gray')
    plt.axis('off')
    plt.title('patterned illumination')
    
    plt.subplot(2, 3, 5) 
    plt.imshow(mov_streak.mean(0), cmap='gray')
    plt.axis('off')
    plt.title('Streaked movie')
    
    plt.subplot(2, 3, 6)
    plt.imshow(A.sum(0), cmap='gray')
    plt.axis('off')
    plt.title('Masks for reconstruction')    
    plt.show()
    
    output = {'streak': mov_streak, 'weighted_masks': A_init, 'streak_masks': A}
    return output

def reconstruction(mov_streak, A, cr=10, size=5, 
                   method='ridge', ridge_alpha=0.1, lasso_alpha=0.0001, positive=False):
    # preparation
    print('reconstruction')
    print(method)
    if method == 'ridge':
        print(ridge_alpha)
    elif method == 'lasso':
        print(lasso_alpha)
    # initialize A,C
    n_masks = cr
    n_comp = A.shape[0]
    nn = n_comp//cr
    A = A / (np.linalg.norm(A, axis=(1, 2), ord='fro')[:, None, None])
    A_mask = A.copy()
    A = A.transpose([1, 2, 0])
    A = A.reshape((-1, n_comp), order='F').astype('float64')
    Y = mov_streak.copy().transpose([1, 2, 0])
    #breakpoint()
    
    # reconstruction, four different methods
    if method == 'nmf':
        T = mov_streak.shape[0]
        C = np.ones((n_comp, T))
        C = C / C.sum(1)[:, None]
        A, C = hals(Y=Y, A=A, C=C, b=None, f=None, bSiz=None, maxIter=5, update_shape=True)
    elif method == 'weighted':
        Y = Y.reshape((-1, Y.shape[-1]), order='F')
        C = A.T@Y
    elif method == 'ridge':
        Y = Y.reshape((-1, Y.shape[-1]), order='F')
        #W = (Y[:, 0]>0).astype(int)
        #Y = (Y - Y.mean()) / Y.std()
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=False, positive=positive)
        ridge.fit(A, Y)        
        C = ridge.coef_.T
    elif method == 'lasso':        
        Y = Y.reshape((-1, Y.shape[-1]), order='F')
        lasso = Lasso(alpha=lasso_alpha, fit_intercept=False, positive=positive)
        lasso.fit(A, Y)
        C = lasso.coef_.T
    A = A.reshape((mov_streak.shape[1], mov_streak.shape[2], n_comp), order='F')
    
    # process the output traces
    C_result = []
    for j in range(nn):
        cc = []
        index = np.array(list(range(j*n_masks, (j+1)*n_masks)))
        for i in index:
            temp = C[i]
            cc.append((temp - temp.mean())/temp.std())
        cc = np.array(cc).reshape(-1, order='F')
        C_result.append(cc)
    C_result = np.array(C_result)
    C_result = np.array([-signal_filter(normalize(c), freq=1/3, fr=400) for c in C_result])
    C_result = normalize(C_result)
    
    # visualization
    plt.figure()
    plt.subplot(1, 3, 1) 
    plt.imshow(mov_streak.mean(0), cmap='gray')
    plt.axis('off')
    plt.title('Streaked movie')
    
    plt.subplot(1, 3, 2)
    plt.imshow(A_mask.sum(0), cmap='gray')
    plt.axis('off')
    plt.title('Masks for reconstruction')

    plt.subplot(1, 3, 3) 
    for j in range(nn):
        plt.plot(normalize(C_result[j, :10000]) + j * 8)
    plt.axis('off')
    plt.plot(range(400), [0] * 400, color='black')
    plt.text(150, -3, '1s', color='black')
    plt.title('Reconstructed signals')
    plt.tight_layout()
    plt.show()
    
    output = {'C_result': C_result, 'traces': C, 'spatial':A}    
    return output

def post_processing(C_result, C_gt):
    result = {}
    nn = C_gt.shape[0]
    # correlation    
    corr = []
    for j in range(nn):
        corr.append(np.corrcoef(C_result[j], C_gt[j])[0, 1])
    
    # spnr
    vpy = np.load(data_folder + '/volpy_403106_3min_1_motion_corrected_adaptive_threshold.npy', allow_pickle=True).item()
    vpy_select = np.array([True] * 23)
    vpy_select[del_overlap] = False
    vpy_select = (vpy['snr'] > 4) * (vpy['num_spikes'][:, 2] > 50) * vpy_select
    select = np.delete((vpy['snr'] > 4) * (vpy['num_spikes'][:, 2] > 50), del_overlap)
    spikes_gt = [v for idx, v in enumerate(vpy['spikes']) if vpy_select[idx] == 1]
    spikes_gt1 = []
    for sp in spikes_gt:
        sp = np.array(sp)
        sp = sp - init_frame
        sp = np.delete(sp, np.where(sp<0)[0])
        spikes_gt1.append(sp)
    spnr_gt, noise_gt = compute_spnr(signals=C_gt[select], spikes=spikes_gt1)
    spnr_result, noise_result = compute_spnr(signals=C_result[select], spikes=spikes_gt1)
    spnr_result_to_gt = spnr_result/spnr_gt
    
    # F1 score
    spikes_ccfm = []
    for c in C_result[select]:
        c_filt, spikes, t_rec, templates, low_spikes, thresh2_normalized = denoise_spikes(c, window_length=8, fr=400,  hp_freq=1/3,  clip=100, threshold_method='adaptive_threshold', 
                           min_spikes=10, pnorm=0.5, threshold=3, do_plot=False)
        spikes_ccfm.append(spikes)       
    
    F1_all = []
    precision_all = []
    recall_all = []
    for s1, s2 in zip(spikes_gt1, spikes_ccfm):
        idx1, idx2 = match_spikes_greedy(s1, s2, max_dist=4)
        F1, precision, recall = compute_F1(s1, s2, idx1, idx2)
        F1_all.append(F1)
        precision_all.append(precision)
        recall_all.append(recall)
    result = {'corr':corr, 'spnr_result_to_gt':spnr_result_to_gt, 'F1':F1_all}
    #
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    j=7
    plt.plot(normalize(C_result[j]) + 8)
    plt.plot(normalize(C_gt[j, :19998]))
    plt.xlim(2000, 2600)
    plt.legend(['Reconstructed', 'Ground truth'], loc=1)
    plt.axis('off')
    plt.plot(range(2000, 2100), [-3] * 100, color='black')
    plt.text(2000, -3, '0.25s', color='black')
    plt.title(f'Example trace')
    #plt.savefig('/home/nel/CODE/compressive_micro/simulation/result/figs/reconstructed_traces_v2.1.pdf')

    # plt.subplot(1, 2, 2)
    # scatter_boxplot([corr, spnr_result_to_gt, F1_all])
    # plt.xticks([1, 2, 3], ["Pearson's r", "SPNR", "F1 score"])
    # plt.ylabel('Value')
    # plt.ylim(0.5, 1)
    # plt.show()
    # plt.subplot(1, 4, 3)
    # scatter_boxplot([spnr_result_to_gt])
    # plt.ylabel("reconstructed SpNR / ground truth SPNR")

    # plt.subplot(1, 4, 4)
    # scatter_boxplot([F1_all])
    # plt.ylabel("F1 score")
    # plt.tight_layout()
#    plt.savefig(save_folder + f'/cr_{cr}_size_{size}_metric.pdf')
    #plt.savefig(save_folder + f'/cr_{cr}_metric1.pdf')
    return result

#%% generate one streak movie
size = 5
for cr in [10]:
    out = generate_streak_movie(mov, mask, init_frame=init_frame, cr=cr, size=size, 
                            poisson=False)
    np.save(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}', out)
    
#%% generate movies with different streak sizes
cr = 10
for size in [3,  7, 9]:
    out = generate_streak_movie(mov, mask, init_frame=init_frame, cr=cr, size=size, 
                            poisson=False)
    np.save(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}', out)
    
#%% generate movies with different compression ratios
size = 5
for cr in [5, 15, 20, 25]:
    out = generate_streak_movie(mov, mask, init_frame=init_frame, cr=cr, size=size, 
                            poisson=False)
    np.save(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}', out)
    
#%% perform reconstruction on movies with different streak sizes
cr = 10
for size in [3, 5, 7, 9]:
    out = np.load(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}.npy', allow_pickle=True).item()
    mov_streak = out['streak']
    A = out['streak_masks']
    
    for method in ['ridge', 'weighted', 'nmf']:
        if method == 'ridge':
            for param in [1e-2, 1e-1, 1, 10]:
                for positive in [False]:
                    out_recon = reconstruction(mov_streak, A, cr=cr, size=size, method=method, 
                                               ridge_alpha=param, positive=positive)
                    np.save(save_folder + f'/scan_speed/{method}_{param}_cr_{cr}_size_{size}_positive_{positive}', out_recon)
        elif method == 'weighted':
            for param in [0]:
                out_recon = reconstruction(mov_streak, A, cr=cr, size=size, method=method)
                np.save(save_folder + f'/scan_speed/{method}_{param}_cr_{cr}_size_{size}', out_recon)
        elif method == 'nmf':
            for param in [0]:
                out_recon = reconstruction(mov_streak, A, cr=cr, size=size, method=method)
                np.save(save_folder + f'/scan_speed/{method}_{param}_cr_{cr}_size_{size}', out_recon)

#%% perform reconstruction on movies with different compression ratio
size = 5
for cr in [5, 10, 15, 20, 25]:
    out = np.load(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}.npy', allow_pickle=True).item()
    mov_streak = out['streak']
    A = out['streak_masks']
    
    for method in ['ridge', 'weighted', 'nmf']:
        if method == 'ridge':
            for param in [1e-2, 1e-1, 1, 10]:
                for positive in [False]:
                    out_recon = reconstruction(mov_streak, A, cr=cr, size=size, method=method, 
                                               ridge_alpha=param, positive=positive)
                    np.save(save_folder + f'/compression_ratio/{method}_{param}_cr_{cr}_size_{size}_positive_{positive}', out_recon)
        elif method == 'weighted':
            for param in [0]:
                out_recon = reconstruction(mov_streak, A, cr=cr, size=size, method=method)
                np.save(save_folder + f'/compression_ratio/{method}_{param}_cr_{cr}_size_{size}', out_recon)
        elif method == 'nmf':
            for param in [0]:
                out_recon = reconstruction(mov_streak, A, cr=cr, size=size, method=method)
                np.save(save_folder + f'/compression_ratio/{method}_{param}_cr_{cr}_size_{size}', out_recon)

    
#%% postprocessing on traces with different streak sizes
cr = 10
corrs = []
spnrs = []
F1s = []
for size in [3, 5, 7, 9]:
    traces = {}
    for method in ['ridge', 'weighted', 'nmf']:
        if method == 'ridge':
            for param in [1]:
                for positive in [False]:
                    out_recon = np.load(save_folder + f'/scan_speed/{method}_{param}_cr_{cr}_size_{size}_positive_{positive}.npy', allow_pickle=True).item()
                    traces[f'{method}_{param}_{positive}'] = out_recon['C_result']
        elif method == 'weighted':
            for param in [0]:
                out_recon = np.load(save_folder + f'/scan_speed/{method}_{param}_cr_{cr}_size_{size}.npy', allow_pickle=True).item()
                traces[f'{method}_{param}'] = out_recon['C_result']
        elif method == 'nmf':
            for param in [0]:
                out_recon = np.load(save_folder + f'/scan_speed/{method}_{param}_cr_{cr}_size_{size}.npy', allow_pickle=True).item()
                traces[f'{method}_{param}'] = out_recon['C_result']
    
    corr = []
    spnr = []
    F1 = []
    print(traces.keys())
    for key in traces.keys():
        result = post_processing(traces[key], C_gt)
        corr.append(result['corr'])
        spnr.append(result['spnr_result_to_gt'])
        F1.append(result['F1'])    
    
    corrs.append(corr)
    spnrs.append(spnr)
    F1s.append(F1)
corrs = np.array(corrs)
spnrs = np.array(spnrs)
F1s = np.array(F1s)
data_scan = {'corrs':corrs, 'spnrs':spnrs, 'F1s':F1s}
#plt.boxplot(np.array(F1s[0]).T)

for key in traces.keys():
    plt.plot(traces[key][7], alpha=0.5)
plt.plot(C_gt[7], alpha=0.5)

#%% Fig 3d
corrs = data_scan['corrs']
spnrs = data_scan['spnrs']
F1s = data_scan['F1s']
xx = np.array([1, 2, 3, 4])
plt.figure(figsize=(14, 4))
ylabels = ['Corr', 'SPNR', 'F1']
cc = ['navy', 'lightsteelblue', 'gray']
for j, metrics in enumerate([corrs, spnrs, F1s]):
    plt.subplot(1, 3, j+1)
    ax = plt.gca()
    for i in range(3):
        bplot = ax.boxplot(metrics[:, i].T, positions=xx+(i-1)*0.2, widths=0.2, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(cc[i])    
    #plt.xticks(xx, ['1.2', '2.0', '2.8', '3.6'])
    plt.xticks(xx, [27, 45, 63, 81])
    plt.ylabel(ylabels[j])
    plt.xlabel('Streak size (px)')
    if j == 0:
        #pass
        #plt.ylim([0.6, 1])
        plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
plt.tight_layout()
plt.savefig('/home/nel/CODE/compressive_micro/simulation/result/figs/performance_scan_speed_v2.2.pdf')

#%% postprocessing on traces with different compression ratios
size = 5
corrs = []
spnrs = []
F1s = []
for cr in [5, 10, 15, 20, 25]:
    traces = {}
    for method in ['ridge', 'weighted', 'nmf']:
        if method == 'ridge':
            for param in [1]:
                for positive in [False]:
                    out_recon = np.load(save_folder + f'/compression_ratio/{method}_{param}_cr_{cr}_size_{size}_positive_{positive}.npy', allow_pickle=True).item()
                    traces[f'{method}_{param}_{positive}'] = out_recon['C_result']
        elif method == 'lasso':
            for param in [1e-6, 1e-5, 1e-4, 1e-3]:
                out_recon = np.load(save_folder + f'/compression_ratio/{method}_{param}_cr_{cr}_size_{size}.npy', allow_pickle=True).item()
                traces[f'{method}_{param}'] = out_recon['C_result']
        elif method == 'weighted':
            for param in [0]:
                out_recon = np.load(save_folder + f'/compression_ratio/{method}_{param}_cr_{cr}_size_{size}.npy', allow_pickle=True).item()
                traces[f'{method}_{param}'] = out_recon['C_result']
        elif method == 'nmf':
            for param in [0]:
                out_recon = np.load(save_folder + f'/compression_ratio/{method}_{param}_cr_{cr}_size_{size}.npy', allow_pickle=True).item()
                traces[f'{method}_{param}'] = out_recon['C_result']
    
    corr = []
    spnr = []
    F1 = []
    print(traces.keys())
    for key in traces.keys():
        result = post_processing(traces[key], C_gt)
        corr.append(result['corr'])
        spnr.append(result['spnr_result_to_gt'])
        F1.append(result['F1'])    
    corrs.append(corr)
    spnrs.append(spnr)
    F1s.append(F1)
    
corrs = np.array(corrs)
spnrs = np.array(spnrs)
F1s = np.array(F1s)
data_cr = {'corrs':corrs, 'spnrs':spnrs, 'F1s':F1s}

plt.boxplot(np.array(F1s[4]).T)

#%% Fig 3c
corrs = data_cr['corrs']
spnrs = data_cr['spnrs']
F1s = data_cr['F1s']

xx = np.array([1, 2, 3, 4, 5])
plt.figure(figsize=(14, 4))
ylabels = ['Corr', 'SPNR', 'F1']
for j, metrics in enumerate([corrs, spnrs, F1s]):
    plt.subplot(1, 3, j+1)
    ax = plt.gca()
    for i in range(3):
        bplot = ax.boxplot(metrics[:, i].T, positions=xx+(i-1)*0.2, widths=0.2, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(cc[i])    
    plt.xticks(xx, ['5', '10', '15', '20', '25'])
    plt.ylabel(ylabels[j])
    plt.xlabel('Compression ratio')
plt.tight_layout()
plt.savefig('/home/nel/CODE/compressive_micro/simulation/result/figs/performance_compression ratio_v2.2.pdf')

#%% Fig 3b
corrs = data_scan['corrs']
spnrs = data_scan['spnrs']
F1s = data_scan['F1s']

xx = np.array([1, 2, 3])
plt.figure(figsize=(6, 6))
ylabels = ['Corr', 'F1', 'SPNR']
for j, metrics in enumerate([corrs, F1s, spnrs]):
    ax = plt.gca()
    bplot = ax.boxplot(metrics[1, :, :].T, positions=[xx[j]-0.2,xx[j],xx[j]+0.2], widths=0.2, patch_artist=True)
    for patch, color in zip(bplot['boxes'], cc):
        patch.set_facecolor(color)    
    plt.xticks(xx, ylabels)
    plt.ylabel('Value')
plt.legend(['Ridge', 'Weighted', 'NMF'], loc='lower right')
plt.tight_layout()
plt.ylim([0.7, 1])
plt.yticks([0.7, 0.8, 0.9, 1.0])
plt.savefig('/home/nel/CODE/compressive_micro/simulation/result/figs/performance_three_metrics_v2.2.pdf')

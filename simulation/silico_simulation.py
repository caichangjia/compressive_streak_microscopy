#!/usr/bin/env python
"""
This script is used to perform simulation of compressive microscopy. 
The simulation uses real voltage imaging datasets as input, simulates targeted 
illumintion and streaking, and performs digitial reconstruction for high
temporal resolution traces afterwards.
author: @caichangjia
"""
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'font.size' : 18.0,
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False, 
                     'xtick.major.size': 7, 
                     'ytick.major.size': 7})
import matplotlib.pyplot as plt
%matplotlib qt

#%%
import logging
import numpy as np
from numpy.linalg import cond
from numpy.random import normal
from utils import load_movie, mov_interpolation, generate_streak_mov, generate_streak_mov_masks, hals, signal_filter, play, normalize#, generate_coded_img
from utils import add_gaussian_noise, save_movie, denoise_spikes
import scipy
from utils import save_movie, compute_spnr, imshow_label, match_spikes_greedy, compute_F1, scatter_boxplot
from skimage.util import random_noise
from sklearn.linear_model import Ridge, Lasso
#from utils import play

#%% load movie
#mov = load_movie('/home/nel/CODE/temporal_optical_encoder/voltage_movie/raw/403106_3min_1.tiff')
data_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Datasets/changjia_compressive_microscope/voltage_movie'
save_folder = '/home/nel/CODE/compressive_micro/simulation/result'
mov = load_movie(data_folder +'/403106_3min_1_motion_corrected.tiff')
mov_raw = mov.copy()
mask = load_movie(data_folder +'/403106_3min_1_motion_corrected_mrcnn_ROIs.hdf5')

#%% generate GT
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
del_overlap = np.array([8, 9, 12, 14, 22])  # delete overlapping neurons
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

#%% functions for perform streak simulation
def generate_streak_movie(mov, mask, init_frame, cr=10, size=5, bg_noise=0.2):
    print(f'compression ratio: {cr}')
    print(f'size: {size}')
    nx = mov.shape[1]
    ny = mov.shape[2] 
    nn = mask.shape[0]
    factor_inter = 5  # interpolation factor

    # compute the average background signal
    bg_frame = mov[init_frame:].mean(0)
    bg_max = (bg_frame[50:60, 55:65]).mean()  
    bg_max /= factor_inter
    bg = bg_max * bg_noise
    
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
    mov_inter = mov_interpolation(mov=mov[init_frame:], factor=factor_inter, method='piecewise_linear')
    mov = mov_inter
    
    # DMD patterned illumination
    print('DMD pattern illumination')
    T_raw = mov.shape[0]
    mov = mov.reshape((T_raw, -1), order='F')
    tmp = mask.reshape((mask.shape[0], -1), order='F').sum(axis=0)
    mov[:, tmp==0] = 0
    mov = mov.reshape((T_raw, nx, ny), order='F')
    
    # add padding, add random gaussian noise
    print('add padding, add background noise for targeted illumination image')   
    padding = 10 * size
    tmp = np.zeros((T_raw, nx+padding, ny))
    tmp[:, :nx, :] = mov
    mov = tmp
    if bg_noise == 0:
        mov_g = mov
    else:
        mov_g = mov + normal(loc=bg, scale=np.sqrt(bg), size=mov.shape)
    
    # generate streak movie with compressed ratio
    print('generate streak movie')
    mov_tmp, m_img = generate_streak_mov(mov_g, cr=cr, size=size)
    
    # add detector noise
    print('add detector noise')
    mov_streak = normal(loc=mov_tmp, scale=np.sqrt(mov_tmp), size=mov_tmp.shape)
        
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
    plt.imshow(mov_g[:3000][:, :nx, :].mean(0), cmap='gray')
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
    
    # data = {'raw':mov_raw[0], 'mask':mask, 
    #         'tg':mov_g[0][:nx, :], 'streak':mov_streak[0]}
    # np.save('/home/nel/CODE/compressive_micro/simulation/result/output/img.npy', data)
    
    output = {'streak': mov_streak, 'weighted_masks': A_init, 'streak_masks': A}
    return output

def reconstruction(mov_streak, A, cr=10, size=5, 
                   method='ridge', reg_auto=True, cond_threshold=10, 
                   ridge_alpha=0.1, lasso_alpha=0.0001, positive=False):
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
        if reg_auto:         # automatic selection of ridge regularizer based on condition number
            ATA = A.T@A    
            print(f'condition number of ATA is: {cond(ATA)}')    
            alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
            conds = []
            for alpha in alphas:
                tmp = ATA + np.diag(np.ones(ATA.shape[0])) * alpha
                conds.append(cond(tmp))
            #np.save(f'/home/nel/CODE/compressive_micro/simulation/result/output/cond_{cr}.npy', conds)
            ind = np.where(np.array(conds) < cond_threshold)[0]
            if len(ind) > 0:
                ridge_alpha = alphas[ind[0]]
                print(f'select ridge alpha: {ridge_alpha}')
            else:
                print('no ridge alpha suitable, use default ridge_alpha')
            #plt.plot(np.log10(alphas), conds)    
        Y = Y.reshape((-1, Y.shape[-1]), order='F')
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=False, positive=positive)
        ridge.fit(A, Y)        
        C = ridge.coef_.T
    elif method == 'lasso':        
        Y = Y.reshape((-1, Y.shape[-1]), order='F')
        lasso = Lasso(alpha=lasso_alpha, fit_intercept=False, positive=positive)
        lasso.fit(A, Y)
        C = lasso.coef_.T
        
    # process the output traces
    C_result = []
    for j in range(nn):
        cc = []
        index = np.array(list(range(j*n_masks, (j+1)*n_masks)))
        for i in index:
            tmp = C[i]
            cc.append((tmp - tmp.mean())/tmp.std())
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

    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # j=7
    # plt.plot(normalize(C_result[j]) + 8)
    # plt.plot(normalize(C_gt[j, :19998]))
    # plt.xlim(2000, 2600)
    # plt.legend(['Reconstructed', 'Ground truth'], loc=1)
    # plt.axis('off')
    # plt.plot(range(2000, 2100), [-3] * 100, color='black')
    # plt.text(2000, -3, '0.25s', color='black')
    # plt.title(f'Example trace')
    # plt.savefig('/home/nel/CODE/compressive_micro/simulation/result/figs/reconstructed_traces_v2.1.pdf')

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

#%% generate single movie
size = 5
for cr in [10]:
    for bg_noise in [0.1, 0.3, 0.5]:
        out = generate_streak_movie(mov, mask, init_frame=init_frame, cr=cr, size=size, bg_noise=bg_noise)
        np.save(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}_noise_{bg_noise}', out)
        
    
#%% generate movies with different scanning speed
cr = 10
bg_noise = 0.1
for size in [5]:
    out = generate_streak_movie(mov, mask, init_frame=init_frame, cr=cr, size=size, bg_noise=bg_noise)
    np.save(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}_noise_{bg_noise}', out)
    
#%% generate movies with different compression ratio
size = 5
bg_noise = 0.1
for cr in [5, 15, 20, 25]:
    out = generate_streak_movie(mov, mask, init_frame=init_frame, cr=cr, size=size, bg_noise=bg_noise)
    np.save(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}_noise_{bg_noise}', out)
    
#%% perform reconstruction on movies with different scanning speed
size = 5
cr = 10
for bg_noise in [0, 0.1, 0.2, 0.3, 0.4]:
    out = np.load(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}_noise_{bg_noise}.npy', allow_pickle=True).item()
    mov_streak = out['streak']
    A = out['streak_masks']
    for method in ['ridge', 'weighted', 'nmf'][1:]:
        out_recon = reconstruction(mov_streak, A, cr=cr, size=size, method=method)
        np.save(save_folder + f'/noise/{method}_reg_auto_cr_{cr}_size_{size}_noise_{bg_noise}', out_recon)

#%%
size = 5
cr = 10
corrs = []
spnrs = []
F1s = []

for bg_noise in [0, 0.1, 0.2, 0.3, 0.4]:
    print(bg_noise)
    traces = {}
    for method in ['ridge', 'weighted', 'nmf']:
        out_recon = np.load(save_folder + f'/noise/{method}_reg_auto_cr_{cr}_size_{size}_noise_{bg_noise}.npy', allow_pickle=True).item()
        traces[f'{method}'] = out_recon['C_result']
        
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
data = {'corrs':corrs, 'spnrs':spnrs, 'F1s':F1s}
np.save('/home/nel/CODE/compressive_micro/simulation/result/output/performance_noise.npy', data)


#%%
cr = 10
bg_noise = 0.1
for size in [3, 5, 7, 9]:
    out = np.load(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}_noise_{bg_noise}.npy', allow_pickle=True).item()
    mov_streak = out['streak']
    A = out['streak_masks']
    
    for method in ['ridge', 'weighted', 'nmf']:
        out_recon = reconstruction(mov_streak, A, cr=cr, size=size, method=method)
        np.save(save_folder + f'/scan_speed/{method}_reg_auto_cr_{cr}_size_{size}_noise_{bg_noise}', out_recon)
    
#%% perform reconstruction on movies with different compression ratio
size = 5
for cr in [5, 10, 15, 20, 25]:
    out = np.load(data_folder + f'/streak_movie/streak_cr_{cr}_size_{size}_noise_{bg_noise}.npy', allow_pickle=True).item()
    mov_streak = out['streak']
    plt.figure()
    plt.imshow(mov_streak[0]); plt.colorbar()
    A = out['streak_masks']
    plt.figure()
    plt.imshow(A.sum(0)); plt.colorbar()

    for method in ['ridge', 'weighted', 'nmf']:
        out_recon = reconstruction(mov_streak, A, cr=cr, size=size, method=method)
        np.save(save_folder + f'/compression_ratio/{method}_reg_auto_cr_{cr}_size_{size}_noise_{bg_noise}', out_recon)

    
#%% postprocessing on traces with different scanning speed
cr = 10
corrs = []
spnrs = []
F1s = []
for size in [3, 5, 7, 9]:
    traces = {}
    for method in ['ridge', 'weighted', 'nmf']:
        out_recon = np.load(save_folder + f'/scan_speed/{method}_reg_auto_cr_{cr}_size_{size}_noise_{bg_noise}.npy', allow_pickle=True).item()
        traces[f'{method}'] = out_recon['C_result']
    
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
data = {'corrs':corrs, 'spnrs':spnrs, 'F1s':F1s}
np.save('/home/nel/CODE/compressive_micro/simulation/result/output/performance_scan.npy', data)

#plt.boxplot(np.array(F1s[0]).T)

# for key in traces.keys():
#     plt.plot(traces[key][7], alpha=0.5)
# plt.plot(C_gt[7], alpha=0.5)

#%% postprocessing on traces with different compression ratio
size = 5
corrs = []
spnrs = []
F1s = []
for cr in [5, 10, 15, 20, 25]:
    traces = {}
    for method in ['ridge', 'weighted', 'nmf']:
        out_recon = np.load(save_folder + f'/compression_ratio/{method}_reg_auto_cr_{cr}_size_{size}_noise_{bg_noise}.npy', allow_pickle=True).item()
        traces[f'{method}'] = out_recon['C_result']
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

data = {'corrs':corrs, 'spnrs':spnrs, 'F1s':F1s}
np.save('/home/nel/CODE/compressive_micro/simulation/result/output/performance_cr.npy', data)

#%%
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 4)
cc = ['navy', 'lightsteelblue', 'gray']

data = np.load('/home/nel/CODE/compressive_micro/simulation/result/output/img.npy', allow_pickle=True).item()
plt.figure(figsize=(16.54, 10))

plt.subplot(gs[0, 0])
plt.imshow(data['raw'], cmap='gray')
[plt.contour(m, levels=[0.98], colors='y')for m in mask]
for i, j in enumerate([0, 7, 10, 14, 15]):#np.arange(0, 18)):#[6, 7, 10, 16, 17]):
    m = mask[j]
    xx = np.mean(np.where(m > 0)[0])
    yy = np.mean(np.where(m > 0)[1])
    plt.text(yy-10, xx, s=f'{i}', color='w')
plt.axis('off')
plt.title('Raw image')
ax = plt.gca()

# plt.subplot(gs[0, 1])
# plt.imshow(data['mask'].sum(0), cmap='gray')
# plt.axis('off')

# plt.subplot(gs[0, 2]) 
# plt.imshow(data['tg'], cmap='gray')
# plt.axis('off')

plt.subplot(gs[0, 1]) 
plt.imshow(data['streak'], cmap='gray')
for i, j in enumerate([0, 7, 10, 14, 15]):#np.arange(0, 18)):#[6, 7, 10, 16, 17]):
    m = mask[j]
    xx = np.mean(np.where(m > 0)[0])
    yy = np.mean(np.where(m > 0)[1])
    plt.text(yy-10, xx+25, s=f'{i}', color='w')
plt.axis('off')
plt.title('Streak image')

#
plt.subplot(gs[0:2, 2:4])
cr = 10
size = 5
bg_noise = 0.1
method = 'ridge'
out_recon = np.load(save_folder + f'/noise/{method}_reg_auto_cr_{cr}_size_{size}_noise_{bg_noise}.npy', allow_pickle=True).item()
C_result = out_recon['C_result']
plt.text(8770, 27, 'Neuron #')
for i, j in enumerate([0, 7, 10, 14, 15]):#np.arange(0, 18)):
    plt.plot(normalize(C_result[j]) + 6*i, color='C0', alpha=1, linewidth=0.8)
    plt.plot(normalize(C_gt[j, :19998]) + 6*i, color='black', alpha=1, linewidth=0.8)
    plt.text(8780, 6*i, f'{i}')
    plt.xlim(8800, 9100)
    plt.legend(['Reconstructed', 'Reference'], loc=1, fontsize=18)
    plt.axis('off')
    plt.plot(range(8800, 8840), [-3] * 40, color='black')
    plt.text(8815, -4, '0.1s', color='black')
    plt.vlines(8795, -0.5, 0.5, color='black', clip_on=False)
    plt.text(8788, -1, '1 unit', rotation='vertical')
plt.tight_layout()
    
#
data = np.load('/home/nel/CODE/compressive_micro/simulation/result/output/performance_cr.npy', allow_pickle=True).item()
corrs, spnrs, F1s = data['corrs'], data['spnrs'], data['F1s']

#plt.figure(figsize=(14, 8))
xx = np.array([1, 2, 3, 4, 5])
ylabels = ['Corr', 'SPNR', 'F1']
for j, metrics in enumerate([corrs, spnrs, F1s]):
    if j == 0:
        plt.subplot(gs[1, 1])
    elif j == 1:
        plt.subplot(gs[2, 0])
    if j == 2:
        plt.subplot(gs[2, 1])
    ax = plt.gca()
    bps = []
    for i in range(3):
        bplot = ax.boxplot(metrics[:, i].T, positions=xx+(i-1)*0.2, widths=0.2, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(cc[i])    
        bps.append(bplot)
    plt.xticks(xx, ['5', '10', '15', '20', '25'])
    plt.ylabel(ylabels[j])
    plt.xlabel('Compression ratio')
    if j == 0:
        plt.yticks([0.3, 0.5, 0.7, 0.9])
        plt.legend([bps[0]["boxes"][0], bps[1]["boxes"][0], bps[2]["boxes"][0]], ['Ridge', 'Weighted', 'NMF'], 
                   fontsize=12)
    if j == 1:
        plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        
        
crs = [5, 10, 15, 20, 25]
xx = np.arange(1, 10, 1)
for i, cr in enumerate(crs):
    data = np.load(f'/home/nel/CODE/compressive_micro/simulation/result/output/cond_{cr}.npy')
    #print(data)
    plt.subplot(gs[1, 0])
    plt.plot(xx, data)
plt.hlines(10, xmin=1, xmax=9, linestyles='dashed', color='black')
plt.legend(crs, fontsize=12)
plt.xticks(xx, [-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([0, 20, 40, 60, 80])
plt.ylabel('Condition number')
plt.xlabel(r'Regularization strength $ log_{10}  \alpha $')


data = np.load('/home/nel/CODE/compressive_micro/simulation/result/output/performance_scan.npy', allow_pickle=True).item()
corrs, spnrs, F1s = data['corrs'], data['spnrs'], data['F1s']
xx = np.array([1, 2, 3, 4])
ylabels = ['Corr', 'SPNR', 'F1']

metrics = corrs
plt.subplot(gs[2, 2])
ax = plt.gca()
for i in range(3):
    bplot = ax.boxplot(metrics[:, i].T, positions=xx+(i-1)*0.2, widths=0.2, patch_artist=True)
    for patch in bplot['boxes']:
        patch.set_facecolor(cc[i])    
plt.xticks(xx, [27, 45, 63, 81])
plt.ylabel(ylabels[0])
plt.xlabel('Streak size (px)')
plt.yticks([0.3, 0.5, 0.7, 0.9])

data = np.load('/home/nel/CODE/compressive_micro/simulation/result/output/performance_noise.npy', allow_pickle=True).item()
corrs, spnrs, F1s = data['corrs'], data['spnrs'], data['F1s']
xx = np.array([1, 2, 3, 4, 5])
labels = ['Corr', 'SPNR', 'F1']
cc = ['navy', 'lightsteelblue', 'gray']
metrics = corrs
plt.subplot(gs[2, 3])
ax = plt.gca()
for i in range(3):
    bplot = ax.boxplot(metrics[:, i].T, positions=xx+(i-1)*0.2, widths=0.2, patch_artist=True)
    for patch in bplot['boxes']:
        patch.set_facecolor(cc[i])    
plt.xticks(xx, [0, 0.1, 0.2, 0.3, 0.4])
plt.ylabel(ylabels[0])
plt.xlabel('Background noise level')
plt.yticks([0.3, 0.5, 0.7, 0.9])
plt.tight_layout()

plt.savefig('/home/nel/CODE/compressive_micro/simulation/result/figs/fig3_v3.0.pdf')



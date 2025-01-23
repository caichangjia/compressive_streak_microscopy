#!/usr/bin/env python
"""
This file is used to perform postprocessing of the compressed movie for beads experiments
@author: @caichangjia
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from match_spikes import match_spikes_greedy, compute_F1
import os
from scipy.signal import find_peaks
from reconstruction import Reconstruction
from skimage.filters import gaussian
from skimage import io
from utils import normalize, pnr, register_translation, compute_lag

mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'font.size' : 18, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False, 
                     'xtick.major.size': 7, 
                     'ytick.major.size': 7})

#%% 
save_dir = r'C:/Users/nico/Desktop/data/fluo_beads_1_13'
save_img_folder = save_dir + '/images'
save_result_folder = save_dir + '/result'
save_stack = save_dir + '/image_stacks'
save_ind = '1'

try:
    os.makedirs(save_result_folder)
    print('create folder')
except:
    print('folder already created')


pixels = io.imread(save_img_folder+'/beads.tif')
pixels0 = io.imread(save_img_folder+'/beads_targeted.tif')
pixels_streak = io.imread(save_dir+'/cr_10_1/cr_10_NDTiffStack.tif') # compression 10, volt 0.08
xy = np.load(save_img_folder+'\\locs.npy')
fr_orig = 400
#cm.movie(pixels_streak, fr=40).save(save_result_folder +'/beads_streak.avi')

#%% compute inferred shifts for galvo
f_shifts = save_dir + f'\\shifts_1\\shifts_NDTiffStack.tif'
m = io.imread(f_shifts)
mm = gaussian(m, channel_axis=0)
for i in range(mm.shape[0]):
    plt.figure()
    plt.imshow(mm[i])
    plt.title(i)
    plt.show()
    
#%%
n = 0
for cr in [5, 10, 15, 20, 25]:
    mmm = mm[n : n + (cr + 1) * 5]
    for j, volt in enumerate([0.02, 0.04, 0.06, 0.08, 0.10]):
            print(cr)
            print(volt)
            shifts = []
            for i in range(cr):
                shifts.append(register_translation(mmm[i + j * (cr + 1)], mmm[(j + 1) * (cr + 1) - 1], upsample_factor=10, max_shifts=(100, 100)))
            shifts = np.array(shifts)    # shifts[0] y axis; shifts[1] x axis
            print(shifts)
            #np.save(save_img_folder+f'/shifts_cr_{cr}_volt_{volt}.npy', shifts)
    n = n + (cr + 1) * 5

#%% Fig a
imgs = []
for d in range(520, 1920, 5):
    filepath = save_stack + f'\\beads_{d}'  + '.tif'
    imgs.append(io.imread(filepath))    
imgs = np.array(imgs)

#%%
plt.figure(figsize=(5, 15))
for i in range(5):
    plt.subplot(5, 1, i+1)
    b = 50
    d = i * 300
    z = -d
    filepath = save_stack + f'\\beads_{d}'  + '.tif'
    p = io.imread(filepath)
    plt.imshow(p[500-b:1600+b, 400-b:2400+b], cmap='gray', vmax=np.percentile(p, 99.8))
    plt.text(500-b, 400-b, f'z={z}um')    
    plt.axis('off')
plt.savefig(save_result_folder+'/3D.pdf')

#%% Fig b
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(pixels, cmap='gray', vmax=np.percentile(pixels, 99.9))
plt.scatter(xy[:, 1], xy[:, 0], s=0.3, c='red')
plt.subplot(1, 2, 2)
plt.imshow(pixels0, cmap='gray', vmax=np.percentile(pixels0, 99.9))
plt.scatter(xy[:, 1], xy[:, 0], s=0.3, c='red')

f_wf = []
f_tg = []
for x, y in xy:
    f_wf.append(pixels[x, y])
    f_tg.append(pixels0[x, y])
f_wf = np.array(f_wf)
f_tg = np.array(f_tg)
print(np.mean(f_wf))
print(np.mean(f_tg))

bg_x = [860, 910]
bg_y = [500, 550]
bg_wf =  pixels[bg_x[0]:bg_x[1], bg_y[0]:bg_y[1]].mean()
bg_tg =  pixels0[bg_x[0]:bg_x[1], bg_y[0]:bg_y[1]].mean()
print(bg_wf)
print(bg_tg)

dff_wf = (f_wf - bg_wf) / bg_wf
dff_tg = (f_tg - bg_tg) / bg_tg
print(dff_wf.mean())
print(dff_tg.mean())

#%%
pixels_st = np.mean(pixels_streak, axis=0)
plt.imshow(pixels_st)

#%%
plt.figure(figsize=(15, 5))
plt.subplot(2, 3, 1)
b = 50
c =(650, 100)
w = (300, 300)
plt.imshow(pixels[500-b:1600+b, 400-b:2400+b], cmap='gray', vmax=np.percentile(pixels, 99.8))
plt.gca().add_patch(Rectangle((c[0], c[1]), 300, 300, edgecolor='black', facecolor='None'))
plt.scatter(xy[:, 1]-(400-b), xy[:, 0]-(500-b), color='red', s=0.5)
#plt.gca().invert_yaxis()
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(pixels0[500-b:1600+b, 400-b:2400+b], cmap='gray', vmax=np.percentile(pixels0, 99.8))
plt.scatter(xy[:, 1]-(400-b), xy[:, 0]-(500-b), color='red', s=0.5)
plt.gca().add_patch(Rectangle((c[0], c[1]), 300, 300, edgecolor='black', facecolor='None'))
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(pixels_st[500-b:1600+b, 400-b:2400+b], cmap='gray', vmax=np.percentile(pixels_st, 99.8))
plt.scatter(xy[:, 1]-(400-b), xy[:, 0]-(500-b), color='red', s=0.5)
plt.gca().add_patch(Rectangle((c[0], c[1]), 300, 300, edgecolor='black', facecolor='None'))
plt.axis('off')

plt.subplot(2, 3, 4, aspect='equal')
plt.imshow(pixels[500-b:1600+b, 400-b:2400+b], cmap='gray', vmax=np.percentile(pixels, 99.8))
#plt.gca().add_patch(Rectangle((c[0], c[1]), 300, 300, edgecolor='red', facecolor='None'))
plt.scatter(xy[:, 1]-(400-b), xy[:, 0]-(500-b), color='red', s=0.5)
#for i in range(len(xy)):
for idx, i in enumerate([74, 15, 30, 36]):
    plt.text(xy[i, 1]-(400-b), xy[i, 0]-(500-b), idx)


plt.xlim([c[0], c[0]+w[0]])
plt.ylim([c[1], c[1]+w[1]])
plt.gca().invert_yaxis()
plt.axis('off')

plt.subplot(2, 3, 5)
#plt.imshow(pixels0[500-b+c[1]:500-b+c[1]+w[1], 400-b+c[0]:400-b+c[0]+w[0]], cmap='gray', vmax=np.percentile(pixels0, 99.8))
plt.imshow(pixels0[500-b:1600+b, 400-b:2400+b], cmap='gray', vmax=np.percentile(pixels0, 99.8))
plt.scatter(xy[:, 1]-(400-b), xy[:, 0]-(500-b), color='red', s=0.5)
plt.xlim([c[0], c[0]+w[0]])
plt.ylim([c[1], c[1]+w[1]])
plt.gca().invert_yaxis()
plt.axis('off')
plt.subplot(2, 3, 6, aspect='equal')
#plt.imshow(pixels_st[500-b+c[1]:500-b+c[1]+w[1], 400-b+c[0]:400-b+c[0]+w[0]], cmap='gray', vmax=np.percentile(pixels_st, 99.8))
plt.imshow(pixels_st[500-b:1600+b, 400-b:2400+b], cmap='gray', vmax=np.percentile(pixels_st, 99.8))
plt.scatter(xy[:, 1]-(400-b), xy[:, 0]-(500-b), color='red', s=0.5)
plt.xlim([c[0], c[0]+w[0]])
plt.ylim([c[1], c[1]+w[1]])
plt.gca().invert_yaxis()
plt.axis('off')
#plt.savefig(save_result_folder+'/fov_v1.1.pdf')

#%% Fig c
plt.figure()
data_to_plot = [dff_wf, dff_tg]
plt.boxplot(data_to_plot, labels=['Widefield', 'Targeted'])
plt.xlabel(None)
plt.ylabel('relative fluorescence intensity')
#plt.title('DFF')
plt.tight_layout()
plt.savefig(save_result_folder+'/wf_targeted.pdf')
plt.show()

#%% reconstruction, load static movie for extracting spatial footpirnts
f_fixed = save_dir + '\\origin_1\\origin_NDTiffStack.tif'
mov_fixed = io.imread(f_fixed)

#%% performance vs cr, weighted and nmf
traces = []
tt = 3
volt = 0.08
for i, cr in enumerate([10, 15, 20, 25]):
    f_streak = save_dir + f'\\cr_{cr}_1\\cr_{cr}_NDTiffStack.tif'
    mov = io.imread(f_streak)
    name = f'\\cr_{cr}_1'
    print(cr)
    print(volt)
    rec = Reconstruction(mov, mov_fixed, locs=xy, cr=cr, volt=volt, base_dir=save_dir, 
                          save_dir=save_dir + name + '\\reconstruction', method='weighted', plot=False)
    traces.append(rec.reconstruct_traces())
traces = np.array(traces)
#np.save('C:/Users/nico/Desktop/data/fluo_beads_1_13/result/cr_traces_weighted.npy', traces)

#%% performance vs cr, ridge and lasso
tt = 3
volt = 0.08

method = 'ridge'
traces = []
for i, cr in enumerate([10, 15, 20, 25]):
    f_streak = save_dir + f'\\cr_{cr}_1\\cr_{cr}_NDTiffStack.tif'
    mov = io.imread(f_streak)
    name = f'\\cr_{cr}_1'
    print(cr)
    print(volt)
    rec = Reconstruction(mov, mov_fixed, locs=xy, cr=cr, volt=volt, base_dir=save_dir, 
                          save_dir=save_dir + name + '\\reconstruction', 
                          method=method, plot=False)
    traces.append(rec.reconstruct_traces())
traces = np.array(traces)    
np.save(save_dir + f'\\result\\cr_traces_ridge_reg_auto.npy', traces)

#%% performance vs volt, weighted and nmf
f_streak = save_dir + '\\volt_cr_10_1\\volt_cr_10_NDTiffStack.tif'
mov = io.imread(f_streak)
name = '\\volt_cr_10_1'

traces = []
tt = 3
tps = 0
cr = 10
method='weighted'
for i, volt in enumerate([0.02, 0.04, 0.06, 0.08, 0.1]):
#for i, volt in enumerate([0.04]):

    print(cr)
    print(volt)
    m = mov[tps : tps + int(tt * fr_orig / cr)]
    rec = Reconstruction(m, mov_fixed, locs=xy, cr=cr, volt=volt, base_dir=save_dir, 
                          save_dir=save_dir + name + '\\reconstruction', method=method, plot=False)
    rec.reconstruct_single_trace(nid=0)
#     traces.append(rec.reconstruct_traces())
    tps = tps + int(tt * fr_orig / cr)
# traces = np.array(traces)
#np.save('C:/Users/nico/Desktop/data/fluo_beads_1_13/volt_cr_10_1/reconstruction/traces_weighted.npy', traces)


#%% performance vs volt, test on ridge and lasso
f_streak = save_dir + '\\volt_cr_10_1\\volt_cr_10_NDTiffStack.tif'
mov = io.imread(f_streak)
name = '\\volt_cr_10_1'
tt = 3
cr = 10
method = 'ridge'
traces = []
tps = 0
for i, volt in enumerate([0.02, 0.04, 0.06, 0.08, 0.1]):
    print(cr)
    print(volt)
    m = mov[tps : tps + int(tt * fr_orig / cr)]
    rec = Reconstruction(m, mov_fixed, locs=xy, cr=cr, volt=volt, base_dir=save_dir, 
                          save_dir=save_dir + name + '\\reconstruction', 
                          method=method, plot=False)
    traces.append(rec.reconstruct_traces())
    tps = tps + int(tt * fr_orig / cr)
traces = np.array(traces)
np.save(f'C:/Users/nico/Desktop/data/fluo_beads_1_13/result/volt_traces_ridge_reg_auto.npy', traces)
   

#%% performance vs intensity, weighted and nmf
f_streak = save_dir + '\\intensity_cr_10_1\\intensity_cr_10_NDTiffStack.tif'
mov = io.imread(f_streak)
name = '\\intensity_cr_10_1'
rec = Reconstruction(mov, mov_fixed, locs=xy, cr=10, volt=0.08, base_dir=save_dir,
                      save_dir=save_dir + name + '\\reconstruction', method='weighted', plot=False)
traces = rec.reconstruct_traces()
traces = traces.reshape((traces.shape[0], 5, -1))
traces = traces.transpose([1, 0, 2])
np.save('C:/Users/nico/Desktop/data/fluo_beads_1_13/result/inten_traces_weighted.npy', traces)

#%% performance vs intensity, ridge and lasso
f_streak = save_dir + '\\intensity_cr_10_1\\intensity_cr_10_NDTiffStack.tif'
mov = io.imread(f_streak)
name = '\\intensity_cr_10_1'
method = 'ridge'
rec = Reconstruction(mov, mov_fixed, locs=xy, cr=10, volt=0.08, base_dir=save_dir,
                      save_dir=save_dir + name + '\\reconstruction', 
                      method=method, plot=False)
traces = rec.reconstruct_traces()
traces = traces.reshape((traces.shape[0], 5, -1))
traces = traces.transpose([1, 0, 2])
np.save('C:/Users/nico/Desktop/data/fluo_beads_1_13/result/'
        + f'inten_traces_ridge_reg_auto.npy', traces)



#%% Fig 4 e, f, g, performance under different LED power, compression ratio, galvo max voltage
#compute three metrics: corr, F1, pnr , only show corr in the paper
intensity_all = np.load(save_img_folder+'/intensity.npy') # intensity matrix contains groundtruth traces
# traces_all = np.load(['C:/Users/nico/Desktop/data/fluo_beads_1_13/result/inten_traces_nmf.npy',
#                       'C:/Users/nico/Desktop/data/fluo_beads_1_13/result/cr_traces_nmf.npy', 
#                       'C:/Users/nico/Desktop/data/fluo_beads_1_13/result/volt_traces_nmf.npy',
#                       'C:/Users/nico/Desktop/data/fluo_beads_1_13/result/volt_traces_weighted.npy', 
#                       'C:/Users/nico/Desktop/data/fluo_beads_1_13/result/inten_traces_weighted.npy'][4])

for method in ['ridge']:
    for test in ['volt', 'cr', 'inten']:
        print(method)
        traces_all = np.load(f'C:/Users/nico/Desktop/data/fluo_beads_1_13/result/{test}_traces_ridge_reg_auto.npy')
        
        results = []
        for i, traces in enumerate(traces_all):
            result = {'corr':[], 'F1':[], 'pr':[], 're':[], 'pnr':[]}    
            if np.any(np.isnan(traces)):   # skip nan
                result['corr'] = [0]*traces_all.shape[1]
                result['F1'] = [0]*traces_all.shape[1]
                result['pr'] = [0]*traces_all.shape[1]
                result['re'] = [0]*traces_all.shape[1]
                result['pnr'] = [0]*traces_all.shape[1]
            else:
                lags = []
                for nid in range(5):
                    lags.append(compute_lag(traces, intensity_all, nid))
                print(lags)
                assert len(np.unique(lags)) == 1
                lag = lags[0]
            
                for nid in range(traces.shape[0]):
                    cc = traces[nid]
                    gt = intensity_all[nid]
                    tp = traces.shape[1]
                    tp_short = gt.shape[0]
                    gt = np.array([gt]*(tp//gt.shape[0]+2)).reshape(-1)
                    gt1 = gt[-lag:-lag+tp]
                    corr = np.corrcoef(normalize(gt1), cc)[0, 1]  # DO NOT APPLY GAUSSIAN FILTER HERE
                    
                    x = normalize(cc)
                    gt_peaks = np.where(gt1>1)[0]
                    peaks, _ = find_peaks(x, height=1.5, distance=2)
                    idx1, idx2 = match_spikes_greedy(gt_peaks, peaks, max_dist=4)
                    F1, precision, recall = compute_F1(gt_peaks, peaks, idx1, idx2)
                    pnr = np.mean(x[gt_peaks])                   
                    
        
                    result['corr'].append(corr)
                    result['F1'].append(F1)
                    result['pr'].append(precision)
                    result['re'].append(recall)
                    result['pnr'].append(pnr)                
                
            results.append(result)
        np.save(save_result_folder+f'/{test}_ridge_reg_auto_v2.0.npy', results)
    
#%%
from matplotlib.gridspec import GridSpec
plt.figure(figsize=(16.54, 4), constrained_layout=True)
gs = GridSpec(1, 4)
plt.subplot(gs[0, 0])

C_result = np.load(f'C:/Users/nico/Desktop/data/fluo_beads_1_13/result/cr_traces_ridge_reg_auto.npy')[0] # cr = 10
intensity = np.load(save_img_folder+'/intensity.npy')
lags = []
for nid in range(5):
    lags.append(compute_lag(C_result, intensity, nid))
print(lags)
assert len(np.unique(lags)) == 1
lag = lags[0]

plt.text(280, 21, 'Neuron #')
for i, j in enumerate([74, 15, 30, 36]):#np.arange(0, 18)):
    gt = intensity[j]
    tp = C_result.shape[1]
    tp_short = gt.shape[0]
    gt = np.array([gt]*(tp//gt.shape[0]+2)).reshape(-1)
    gt1 = gt[-lag:-lag+tp]

    plt.plot(normalize(C_result[j]) + 6*i, color='C0', alpha=1)
    plt.plot(normalize(gt1) + 6*i, color='black', alpha=0.5)
    plt.text(285, 6*i, f'{i}')
    plt.xlim([300, 370])
    plt.axis('off')
    plt.plot(range(300, 320), [-3] * 20, color='black')
    plt.text(302, -6, '0.05s', color='black')
    plt.vlines(298, -0.5, 0.5, color='black', clip_on=False)
    plt.text(292, -2, '1 unit', rotation='vertical')
plt.legend(['Reconstructed (Ridge)', 'Reference'], fontsize=14, loc='lower right')

# ax = plt.gca()
# pos = ax.get_position()  # Get the original position
# ax.set_position([pos.x0, pos.y0, pos.width*1.2, pos.height])  # Adjust position



data = {}
ylabels = ['Corr', 'SPNR', 'F1']
cc = ['navy', 'lightsteelblue', 'gray']

for ii, test in enumerate(['inten', 'cr', 'volt']):
    if ii == 0:
        plt.subplot(gs[0, 1])
    if ii == 1:
        plt.subplot(gs[0, 2])
    if ii == 2:
        plt.subplot(gs[0, 3])
    metric = 'corr'
        
    for method in ['ridge', 'weighted', 'nmf']:
        if method == 'ridge':
            results = np.load(save_result_folder+f'/{test}_{method}_reg_auto_v2.0.npy', allow_pickle=True)
        else:
            results = np.load(save_result_folder+f'/{test}_{method}_v1.0.npy', allow_pickle=True)
        rr = np.array([r[metric] for r in results])
        data[f'{method}'] = rr
        
    xx = np.arange(1, data['ridge'].shape[0]+1)
    ax = plt.gca()
    bps = []
    for i, key in enumerate(data.keys()):
        rr = data[key]
        #if key == 'nmf':   # remove that column of nmf due to fail to reconstruct
        #    rr[0] = np.nan
        bplot = ax.boxplot(rr.T, positions=xx+(i-1)*0.2, widths=0.2, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(cc[i])    
        bps.append(bplot)
            
    plt.ylabel('Correlation')
    if test == 'inten':
        plt.xlabel('LED driver voltage (V)')
        plt.xticks([1, 2, 3, 4, 5], ['0.25', '0.5', '0.75', '1.0', 1.25])
        plt.yticks([0.3, 0.5, 0.7, 0.9])
        plt.legend([bps[0]["boxes"][0], bps[1]["boxes"][0], bps[2]["boxes"][0]], ['Ridge', 'Weighted', 'NMF'], fontsize=14)
    elif test == 'cr':
        plt.xlabel('Compression ratio')
        plt.xticks([1, 2, 3, 4], [10, 15, 20, 25])
    elif test == 'volt':
        plt.xlabel('Galvo maximum \n voltage (mV)')
        plt.xticks([1, 2, 3, 4, 5], [20, 40, 60, 80, 100])
    
plt.tight_layout()
plt.savefig('C:/Users/nico/Desktop/data/fluo_beads_1_13/result/figs/performance_v3.0.pdf')

#%% Fig 4e extended
data = {}
cc = ['navy', 'lightsteelblue', 'gray']
plt.figure(figsize=(14, 4))
for ii, mid in enumerate([0, 1, -1]):
    plt.subplot(1, 3, ii+1)
    metrics = ['corr', 'F1', 'pr', 're', 'pnr']
    metric = metrics[mid]
    print(metric)
    
    for method in ['ridge', 'weighted', 'nmf']:
        if method == 'ridge':
            results = np.load(save_result_folder+f'/inten_{method}_reg_auto_v2.0.npy', allow_pickle=True)
            rr = np.array([r[metric] for r in results])
            data[f'{method}'] = rr
        else:
            results = np.load(save_result_folder+f'/inten_{method}_v1.0.npy', allow_pickle=True)
            rr = np.array([r[metric] for r in results])
            data[f'{method}'] = rr
       
    
    xx = np.array([1, 2, 3, 4, 5])
    ax = plt.gca()
    for i, key in enumerate(data.keys()):
        rr = data[key]
        bplot = ax.boxplot(rr.T, positions=xx+(i-1)*0.2, widths=0.2, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(cc[i])    
    plt.ylabel(metric)
    plt.xlabel('LED amplitude')
    plt.xticks([1, 2, 3, 4, 5], ['0.25', '0.5', '0.75', '1.0', 1.25])
    plt.tight_layout()
    #plt.savefig('C:/Users/nico/Desktop/data/fluo_beads_1_13/result/figs/performance_LED_amplitude_v2.0.pdf')
    
#%% Fig 4f extended
data = {}
plt.figure(figsize=(14, 4))
for ii, mid in enumerate([0, 1, -1]):
    plt.subplot(1, 3, ii+1)
    metrics = ['corr', 'F1', 'pr', 're', 'pnr']
    metric = metrics[mid]
    print(metric)
    
    for method in ['ridge', 'weighted', 'nmf']:
        if method == 'ridge':
            results = np.load(save_result_folder+f'/cr_{method}_reg_auto_v2.0.npy', allow_pickle=True)
        else:
            results = np.load(save_result_folder+f'/cr_{method}_v1.0.npy', allow_pickle=True)
        rr = np.array([r[metric] for r in results])
        data[f'{method}'] = rr

    xx = np.array([1, 2, 3, 4])
    ax = plt.gca()
    for i, key in enumerate(data.keys()):
        rr = data[key]
        bplot = ax.boxplot(rr.T, positions=xx+(i-1)*0.2, widths=0.2, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(cc[i])    
    #plt.xticks(xx, ['5', '10', '15', '20', '25'])
    plt.ylabel(metric)
    plt.xlabel('Compression Ratio')
    plt.xticks([1, 2, 3, 4], [10, 15, 20, 25])
    plt.tight_layout()
    #plt.savefig('C:/Users/nico/Desktop/data/fluo_beads_1_13/result/figs/performance_compression_ratio_v2.0.pdf')
    
#%% Fig 4g extended
data = {}
xx = np.array([1, 2, 3, 4, 5])
ylabels = ['Corr', 'SPNR', 'F1']
cc = ['navy', 'lightsteelblue', 'gray']

plt.figure(figsize=(14, 4))

for ii, mid in enumerate([0, 1, -1]):
    plt.subplot(1, 3, ii+1)
    metrics = ['corr', 'F1', 'pr', 're', 'pnr']
    metric = metrics[mid]
    print(metric)
    
    #method=['nmf', 'weighted'][0]
    for method in ['ridge', 'weighted', 'nmf']:
        if method == 'ridge':
            results = np.load(save_result_folder+f'/volt_{method}_reg_auto_v2.0.npy', allow_pickle=True)
        else:
            results = np.load(save_result_folder+f'/volt_{method}_v1.0.npy', allow_pickle=True)
        rr = np.array([r[metric] for r in results])
        data[f'{method}'] = rr
            
    ax = plt.gca()
    for i, key in enumerate(data.keys()):
        rr = data[key]
        if mid == 1:
            if key == 'nmf':   # remove that column of nmf due to fail to reconstruct
                rr[0] = np.nan
        bplot = ax.boxplot(rr.T, positions=xx+(i-1)*0.2, widths=0.2, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(cc[i])    
    #plt.xticks(xx, ['5', '10', '15', '20', '25'])
    plt.ylabel(metric)
    #plt.xlabel('Compression ratio')
    plt.xlabel('Galvo maximum voltage')
    plt.xticks([1, 2, 3, 4, 5], ['0.02', '0.04', '0.06', '0.08', '0.1'])
    plt.tight_layout()
#plt.savefig('C:/Users/nico/Desktop/data/fluo_beads_1_13/result/figs/performance_galvo_voltage_v2.0.pdf')

#%% Fig e, f, g 
metrics = ['corr', 'F1', 'pr', 're', 'pnr']
metric = metrics[1]
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
results = np.load(save_result_folder+'/inten_v1.0.npy', allow_pickle=True)
rr = np.array([r[metric] for r in results])
plt.boxplot(rr.T)
plt.ylabel(metric)
plt.xlabel('LED amplitude')
plt.xticks([1, 2, 3, 4, 5], ['0.25', '0.5', '0.75', '1.0', '1.25'])
plt.ylim([0, 1])

plt.subplot(1, 3, 2)
results = np.load(save_result_folder+f'/cr_v1.0.npy', allow_pickle=True)
rr = np.array([r[metric] for r in results])
plt.boxplot(rr[:-1].T)
plt.ylabel(metric)
plt.xlabel('Compression ratio')
plt.xticks([1, 2, 3], [10, 15, 20])
plt.ylim([0, 1])


plt.subplot(1, 3, 3)
method=['nmf', 'weighted'][0]
results = np.load(save_result_folder+f'/volt_{method}_v1.0.npy', allow_pickle=True)
#plt.figure()
rr = np.array([r[metric] for r in results])
plt.boxplot(rr[1:].T)
plt.ylabel(metric)
plt.xlabel('Galvo maximum voltage (V)')
plt.xticks([1, 2, 3, 4], ['0.04', '0.06', '0.08', '0.10'])
#plt.ylim([0, 1])

plt.tight_layout()
#plt.savefig(save_result_folder+f'/performance_v1.2_{metric}.pdf')


#!/usr/bin/env python
"""
This file is used to perform postprocessing of the streak movie for the fish experiment.
@author: @caichangjia
"""
import cv2
#import caiman as cm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from skimage import io
from utils import normalize, register_translation, apply_shifts_dft, high_pass_filter_space, signal_filter
from scipy.ndimage import gaussian_filter1d as gf
from reconstruction import ReconstructionFish

mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'font.size' : 18, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False, 
                     'xtick.major.size': 7, 
                     'ytick.major.size': 7})

#%%
#save_dir = r'C:/Users/nico/Desktop/data/zebrafish_5_7'
save_dir = r'C:/Users/nico/Desktop/data/zebrafish_7_20_1'
save_img_folder = save_dir + '/images'
save_result_folder = save_dir + '/result'
save_stack = save_dir + '/image_stacks'
save_ind = '1'

try:
    os.makedirs(save_result_folder)
    print('create folder')
except:
    print('folder already created')
fr_orig = 400
#cr = 10
#volt = 0.08

#%% Estimate shifts at different galvo location
#f_shifts = 'C:/Users/nico/Desktop/data/zebrafish_5_7/shifts_2/shifts_NDTiffStack.tif'
f_shifts = 'C:/Users/nico/Desktop/data/zebrafish_7_20_1/shifts_1/shifts_NDTiffStack.tif'
m = io.imread(f_shifts)
mm = m.astype(np.float64)
mm = mm[:, 100:700, 500:900]

for i in range(mm.shape[0]):
    plt.figure()
    plt.imshow(mm[i])
    plt.title(i)
    plt.show()

#%% 
n = 0
new_imgs = []
for cr in [10, 15, 20]:
    mmm = mm[n : n + (cr + 1) * 3]
    for j, volt in enumerate([0.04, 0.06, 0.08]):
            print(cr)
            print(volt)
            shifts = []
            for i in range(cr):
                img = high_pass_filter_space(mmm[i + j * (cr + 1)], gSig_filt=(3,3))
                shift, src_freq, phasediff = register_translation(img, mmm[(j + 1) * (cr + 1) - 1], upsample_factor=10, max_shifts=(100, 100))
                shifts.append(shift)
                new_img = apply_shifts_dft(mmm[i + j * (cr + 1)], [-shift[0], -shift[1]], diffphase=phasediff, is_freq=False, border_nan=True)
                new_imgs.append(new_img)   
            shifts = np.array(shifts)    # shifts[0] y axis; shifts[1] x axis
            print(shifts)
            #np.save(save_img_folder+f'/shifts_cr_{cr}_volt_{volt}.npy', shifts)
    n = n + (cr + 1) * 3
new_imgs = np.array(new_imgs)


for i in range(new_imgs.shape[0]):
    if i < 12:
        plt.figure()
        plt.imshow(new_imgs[i])
        plt.title(i)
        plt.show()

#%% Load widefield, targeted movie, remove neurons suffered from scattering
wf = io.imread('C:/Users/nico/Desktop/data/zebrafish_7_20_1/wf_movie_3/wf_movie_NDTiffStack.tif')
tg = io.imread('C:/Users/nico/Desktop/data/zebrafish_7_20_1/tg_movie_4/tg_movie_NDTiffStack.tif')
xy = np.load(save_img_folder+'\\locs_wf_v3.0.npy')
xy_delete = xy[np.array([2, 3, 9, 10, 30, 34])]
xy = np.delete(xy, [2, 3, 9, 10, 30, 34], axis=0)

#%%
lim_y = [600, 1400]#[::-1]
lim_x = [1200, 1800]
wf_save = wf[:, lim_y[0]:lim_y[1], lim_x[0]:lim_x[1]]
tg_save = tg[:, lim_y[0]:lim_y[1], lim_x[0]:lim_x[1]]
# cm.movie(wf_save, fr=50).resize(1, 1, 0.1).save(save_result_folder +'/wf_movie_3_resize_0.1.avi')
# cm.movie(tg_save, fr=50).save(save_result_folder +'/tg_movie_4_50Hz.avi')
#%%
mean_img = wf.mean(0)
std_img = wf.std(0)
tg_mean_img = tg.mean(0)
tg_std_img = tg.std(0)

#%%
mean_img1 = mean_img[lim_y[0]:lim_y[1], lim_x[0]:lim_x[1]]
std_img1 = std_img[lim_y[0]:lim_y[1], lim_x[0]:lim_x[1]]
tg_mean_img1 = tg_mean_img[lim_y[0]:lim_y[1], lim_x[0]:lim_x[1]]
streak_mean_img1 = streak_mean_img[lim_y[0]:lim_y[1], lim_x[0]:lim_x[1]]


#%% plot detected neurons
plt.subplot(1, 2, 1)
plt.imshow(mean_img, cmap='gray')
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
plt.title('Widefield mean img')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(std_img, cmap='gray')
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
plt.title('Std img')
plt.axis('off')
for i in range(36):
    plt.text(xy[i, 1], xy[i, 0], s=f'{i}', color='red')

#%% 
spatial = []
for xx in xy:
    radius = 10
    temp = np.zeros(wf.shape[1:])
    cv2.circle(temp, [xx[1], xx[0]], radius, 1, -1)
    spatial.append(temp)
spatial = np.array(spatial)
#plt.imshow(spatial.sum(0))

plt.figure()
plt.imshow(std_img, cmap='gray')
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
plt.imshow(spatial.sum(0), alpha=0.2)
plt.title('Std img')
plt.axis('off')

# #%%
# sp = spatial.copy().transpose([1, 2, 0]).reshape((-1, spatial.shape[0]))
# wf1 = wf.reshape((wf.shape[0], -1))
# wf_trace = wf1@sp
# wf_trace = wf_trace.T
# wf_trace_n = normalize(wf_trace)

#%% Fig 5a
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.imshow(mean_img1, cmap='gray', vmax=np.percentile(mean_img1, 99.9))
plt.scatter(xy[:, 1]-lim_x[0], xy[:, 0]-lim_y[0], color='red', alpha=0.5, s=10, marker='o', label='Preprocessed')
plt.scatter(xy_delete[:, 1]-lim_x[0], xy_delete[:, 0]-lim_y[0], color='red', alpha=0.5, s=10, marker='x', label='Removed')

plt.plot([0, 60], [5, 5], color='w')
plt.title('Widefield mean img')
plt.legend()
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(std_img1, cmap='gray', vmax=np.percentile(std_img1, 99.9))
plt.scatter(xy[:, 1]-lim_x[0], xy[:, 0]-lim_y[0], color='red', alpha=0.5, s=10, marker='o')
plt.scatter(xy_delete[:, 1]-lim_x[0], xy_delete[:, 0]-lim_y[0], color='red', alpha=0.5, s=10, marker='x')
plt.title('widefield std img')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(tg_mean_img1, cmap='gray', vmax=np.percentile(tg_mean_img1, 99.9))
plt.scatter(xy[:, 1]-lim_x[0], xy[:, 0]-lim_y[0], color='red', alpha=0.5, s=10, marker='o')
plt.scatter(xy_delete[:, 1]-lim_x[0], xy_delete[:, 0]-lim_y[0], color='red', alpha=0.5, s=10, marker='x')
plt.title('Targeted mean img')
for idx, i in enumerate([2, 3, 4, 5, 6, 8, 14]):
    plt.text(xy[i, 1]-lim_x[0], xy[i, 0]-lim_y[0], idx)

plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(streak_mean_img1, cmap='gray', vmax=np.percentile(streak_mean_img1, 99.9))
plt.scatter(xy[:, 1]-lim_x[0], xy[:, 0]-lim_y[0], color='red', alpha=0.5, s=10, marker='o')
plt.scatter(xy_delete[:, 1]-lim_x[0], xy_delete[:, 0]-lim_y[0], color='red', alpha=0.5, s=10, marker='x')
for i, idx in enumerate([2, 3, 4, 5, 6, 8, 14]):
    plt.text(xy[idx, 1]-lim_x[0], xy[idx, 0]-lim_y[0], s=f'{i}')
plt.title('Streak mean img')
plt.axis('off')
plt.tight_layout()
#plt.savefig(save_result_folder+'/summary_imgs_20Hz_volt_0.04_v3.0.pdf')

#%% reconstruction
tt = 60 # time 
tps = 0
cr = 10 # compression ratio
volt = 0.04 # galvo mirror voltage

#f_streak  = save_dir + '/streak_20hz_4/streak_20hz_NDTiffStack.tif'
f_streak = save_dir + '/streak_40hz_1/streak_40hz_NDTiffStack.tif'

mov = io.imread(f_streak)
streak_mean_img = mov.mean(0)
mov_fixed = tg.copy()
name = f'\\cr_{cr}_1'
print(cr)
print(volt)

data = {}
#methods = ['weighted', 'nmf']
methods = ['ridge']
for method in methods:
    data[method] = {}

for method in methods:
    method_name = method
    print(f'Using method: {method_name}')
    rec = ReconstructionFish(mov, mov_fixed, locs=xy, cr=cr, volt=volt, base_dir=save_dir, 
                          save_dir=save_dir + name + '\\reconstruction', 
                          method=method_name, plot=False)
    #rec.reconstruct_single_trace(nid=1)
    output = rec.reconstruct_traces()
    rec_trace = np.array([out[0] for out in output])
    data[method]['C_result'] = rec_trace
    data[method]['C'] = np.array([out[1] for out in output])
    data[method]['A'] = np.array([out[2] for out in output])
    data[method]['weighted_masks'] = np.array([out[3] for out in output])
    data[method]['small_masks'] = np.array([out[4] for out in output])
    data[method]['C_gt'] = np.array([out[5] for out in output])
    rec_trace_n = normalize(rec_trace)
    for idx in range(len(rec_trace_n)):
        rec_trace_n[idx] = gf(rec_trace_n[idx], sigma=1)
    data[method]['C_result_filtered'] = rec_trace_n    

for method in methods:
    np.save(f'C:/Users/nico/Desktop/data/zebrafish_7_20_1/result/output/{method}_40Hz_v3.0.npy', data[method])
    
#%%
data = {}
methods = ['ridge', 'weighted', 'nmf']
for method in methods:
    d = np.load(f'C:/Users/nico/Desktop/data/zebrafish_7_20_1/result/output/{method}_20Hz_v3.0.npy', allow_pickle=True).item()
    data[method] = d
    
#%% check trace of one neuron
ii = 14
#for ii in range(36):
plt.figure(figsize=(6, 8))
methods = np.array(['ridge'])#[np.array([0, 2])]
for idx, method in enumerate(methods):
    print(method)
    #plt.plot(data[method]['C_result'][ii]+idx*0, alpha=0.5)
    plt.plot(normalize(data[method]['C_result'][ii])+idx*0, alpha=0.5)
    plt.plot(np.arange(1, 12001, 10), normalize(data[method]['C_gt'][ii])+idx*0, alpha=0.5)
plt.legend(methods)
plt.title(f'neuron:{ii}')
plt.show()    

    
#%% the traces are normalized later for visualization
method = 'ridge'
rec_trace_n = data[method]['C_result_filtered']
trace_gt = data[method]['C_gt']
trace_hp = []
for tr in rec_trace_n:
    tr_filtered = signal_filter(tr, freq=1, fr=200)
    trace_hp.append(tr_filtered)
    plt.figure()
    plt.plot(tr); plt.plot(tr_filtered-2); plt.plot(tr-tr_filtered)
    print(tr_filtered.std())
trace_hp = np.array(trace_hp)
hp_signal = trace_hp.std(1)

#%% Fig 5c
from matplotlib.patches import Rectangle
plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1)
fr = 20
if fr == 40:
    x = np.arange(0, 60, 0.0025)
    xx = np.arange(0, 60, 0.025)
    st = 35 
else:
    x = np.arange(0, 60, 0.005)
    xx = np.arange(0.025, 60+0.025, 0.05)
    
    st = 40
    index = np.array([2, 3, 4, 5, 6, 8, 14])
    #x_gt = np.arange(0, 60, 0.05)

for idx, t in enumerate(rec_trace_n[index]):
    #t = t / hp_signal[index][idx]
    t = normalize(t)
    gt = trace_gt[index][idx]
    gt = normalize(gt)
    plt.plot(x, t + idx*5, linewidth=0.5, alpha=1, color='C0')
    plt.plot(xx, gt + idx*5, linewidth=0.5, alpha=1, color='black')
    plt.text(-7, 5*idx, f'{idx}')


rect = Rectangle([st, -3], 10, 40, alpha=1, color='gray', fc = 'none', lw = 2, linestyle='dashed')
ax = plt.gca()
ax.add_patch(rect)
ax.spines['left'].set_visible(False)
#plt.ylabel('Normalized fluorescence')q
plt.legend(['Reconstructed', 'Reference'], fontsize=14)
plt.xlabel('Time (s)')
plt.yticks([])
plt.xlim([0, 60])
plt.text(-12, 40, 'Neuron #')
plt.vlines(-1, -0.5, 0.5, color='black', clip_on=False)
plt.text(-4, -5, '1 unit', rotation='vertical')
#plt.title('fluorescence traces plot')

#
plt.subplot(2, 1, 2)
for idx, t in enumerate(rec_trace_n[index]):
    #t = t / hp_signal[index][idx]
    t = normalize(t)
    gt = trace_gt[index][idx]
    gt = normalize(gt)
    plt.plot(x, t + idx*5, linewidth=0.5, alpha=1, color='C0')
    plt.plot(xx, gt + idx*5, linewidth=0.5, alpha=1, color='black')
    plt.text(st-7/6, 5*idx, f'{idx}')

rect = Rectangle([st, -3], 10, 40, alpha=1, color='gray', fc = 'none', lw = 2, linestyle='dashed')
ax = plt.gca()
ax.add_patch(rect)
ax.spines['left'].set_visible(False)
plt.xlim(st, st+10)
plt.yticks([])
plt.xlabel('Time (s)')
plt.text(st-12/6, 40, 'Neuron #')
plt.vlines(st-1/6, -0.5, 0.5, color='black', clip_on=False)
plt.text(st-4/6, -1, '1 unit', rotation='vertical')

plt.tight_layout()
#plt.savefig(save_result_folder+'/traces_40Hz_volt_0.04_v1.0.pdf')
plt.savefig(save_result_folder+f'/traces_20hz_volt_0.04_v3.0.pdf')

#%% Fig 5b
plt.figure()
for i in range(48):
    if i <= 35:
        plt.subplot(4, 12, i+1)
        d = data['ridge_1']['small_masks'][i]
        plt.imshow(d/(d.max()-d.min()))
        plt.axis('off')
    if i == 36:
        plt.subplot(4, 12, i+1)
        plt.imshow(d/(d.max()-d.min()))
        plt.axis('off')
        plt.colorbar()
#plt.tight_layout()
plt.savefig(save_result_folder+'/spatials_20Hz_volt_0.04_v2.0.pdf')

#%%
streak_save_20 = mov[:, lim_y[0]:lim_y[1], lim_x[0]:lim_x[1]]
cm.movie(streak_save_20, fr=200).save(save_result_folder +'/streak_movie_20_4_200Hz.avi')

#%%
streak_save_40 = mov[:, lim_y[0]:lim_y[1], lim_x[0]:lim_x[1]]
cm.movie(streak_save_40, fr=400).save(save_result_folder +'/streak_movie_40_1_400Hz.avi')


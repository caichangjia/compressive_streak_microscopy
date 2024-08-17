#!/usr/bin/env python
"""
This file is used to perform beads experiment. It detects beads, generates DMD pattern for targeted 
illumination, and acquires streak movies with galvo on. 
@author: @caichangjia
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pycromanager import Core, Acquisition, multi_d_acquisition_events
from skimage import io
from skimage.feature import peak_local_max
from skimage.filters import gaussian
import tifffile
from utils import random_numbers_with_min_distance, generate_intensity_matrix, play, generate_bmp_file, map_to_DMD

#%%
#date = '5_2_10x'
#save_dir = r'C:/Users/nico/Desktop/data/beads_' + date
date = '8_2'
#save_dir = r'C:/Users/nico/Desktop/data/zebrafish_' + date
save_dir = r'C:/Users/nico/Desktop/data/test_8_2'
fr_orig = 400

save_img_folder = save_dir + '/images'
save_dmd_folder = save_dir+'/DMD_pattern'
try:
    os.makedirs(save_dmd_folder)
    print('create folder')
except:
    print('folder already created')

try:
    os.makedirs(save_img_folder)
    print('create folder')
except:
    print('folder already created')

#%% capture images at different galvo mirror angles to infer shifts 
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][0]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][2]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 24.75
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)

#%%
save_name = 'shifts'
tp = 0
volts = [0.02, 0.04, 0.06, 0.08, 0.10]
for cr in [5, 10, 15, 20, 25]:
    tp = tp + (cr + 1) * 1 * len(volts)
with Acquisition(directory=save_dir, name=save_name, show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=tp, time_interval_s=0)
    acq.acquire(events)

#%% setup pycromanager
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][1]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 20
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)

#%% capture a widefield image
core.snap_image()
tagged_image=core.get_tagged_image()
pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'],tagged_image.tags['Width']])
filepath = save_img_folder + '\\beads'  + '.tif'
tifffile.imwrite(filepath,pixels)
i1 = io.imread(save_img_folder+'/beads.tif')
plt.figure()
plt.imshow(pixels, vmax=np.percentile(pixels, 99.99), cmap='gray')
plt.savefig(save_img_folder+'/beads.png')

#%% gaussian filter image, gaussian filter is a must for detecting beads centers accurately
im = pixels.copy()
im_gaussian = gaussian(im, sigma=1)
im_gaussian /= im_gaussian.max()
plt.subplot(1, 2, 1)
plt.imshow(im, vmax=np.percentile(im, 99.99))
plt.title('raw img')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(im_gaussian, vmax=np.percentile(im_gaussian, 99.99))
plt.colorbar()
plt.title('gaussian filtered img')
plt.tight_layout()

#%% detect beads through finding local maximum, generate pattern for tageted illumination on the camera
xy = peak_local_max(im_gaussian, min_distance=30, num_peaks=100, threshold_abs=0.4, exclude_border=False)
img1 = np.zeros(im.shape)
radius = 15
for xx in xy:
    cv2.circle(img1, [xx[1], xx[0]], radius, 1, -1)
print(len(xy))
#np.save(save_img_folder+'/locs_raw.npy', xy)

#%% remove beads based on their distance parallel/perpendicular to streak direction (optional)
# centers = xy.copy()
# min_dist_p = 45    # the min streak size for 0.04, 0.06, 0.08 is [16, 23, 32] *sqrt(2) + 20
# min_dist_v = 10 # neuron diameter estimated as 20 px
# num = centers.shape[0]
# dis = np.zeros((num, num, 2))  # first dimension parallel to the streak direction, second dimension vertical
# for i in range(num):
#     for j in range(i+1, num):
#         dis[i, j, 1] = np.sqrt(2) / 2 * np.abs(centers[i][0] - centers[j][0] - (centers[i][1] - centers[j][1]))
#         dis[i, j, 0] = np.sqrt(2) / 2 * np.abs(centers[i][0] - centers[j][0] + (centers[i][1] - centers[j][1]))

# # select neurons that are above min distance away from other neurons
# selected = np.array([0])
# for i in range(1, num):
#     if sum((dis[:i ,i, 0][selected] < min_dist_p) & (dis[:i ,i, 1][selected] < min_dist_v)) == 0:
#         selected = np.append(selected, i)
#         print(selected)            
# print(len(selected))        

#%%
plt.subplot(1, 2, 1)
plt.imshow(im, vmax=np.percentile(im, 99.99), cmap='gray')
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
plt.title('raw img')
plt.subplot(1, 2, 2)
plt.imshow(im_gaussian, vmax=np.percentile(im_gaussian, 99.99), cmap='gray'); plt.colorbar()
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
plt.title('gaussian img')
plt.tight_layout()
#plt.savefig(save_img_folder + '\\detected_peaks_neurons.png')

#%% load calibration result
reg1, reg2 = np.load(save_dir + '\\calib\\regression_result.npy', allow_pickle=True)

#%% predict using camera pixels
pred = map_to_DMD(img1, reg1, reg2)
assert pred[:, 0].max() < 1080 and pred[:, 0].min() > 0
assert pred[:, 1].max() < 1920 and pred[:, 1].min() > 0

#%% remove points outside DMD boundary
pred = pred[(pred[:, 0] < 1080) & (pred[:, 0] > 0)]
pred = pred[(pred[:, 1] < 1920) & (pred[:, 1] > 0)]

#%% create targeted illumination pattern on DMD
img = generate_bmp_file(pred, dev=[0, 0])  
img.save(save_dmd_folder + f'/all_beads_radius_{radius}.bmp')

#%% capture targeted illumination pattern
core.snap_image()
tagged_image=core.get_tagged_image()
pixels1 = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'],tagged_image.tags['Width']])
filepath = save_img_folder + '\\beads_targeted_raw'  + '.tif'
tifffile.imwrite(filepath,pixels1)
plt.figure()
plt.imshow(pixels1, vmax=np.percentile(pixels1, 99), cmap='gray')
plt.savefig(save_img_folder+'/beads_targeted_raw.png')

#%%
im1 = io.imread(save_img_folder+'/beads_targeted_raw.tif')
im1_gaussian = gaussian(im1, sigma=1)
im1_gaussian /= im1_gaussian.max()

plt.subplot(1, 2, 1)
plt.imshow(im, vmax=np.percentile(im, 99.99), cmap='gray')
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
plt.title('raw img')
#plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(im1, vmax=np.percentile(im1, 99.99), cmap='gray')
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
#plt.colorbar()
plt.title('targeted illumination img')

#%% remove beads that are out-of-focus
nid = 0
num_pixels = []
for nid in range(len(xy)):
    sz = 15 # size of the region
    avg_x = [xy[nid, 0]-sz, xy[nid, 0]+sz]
    avg_y = [xy[nid, 1]-sz, xy[nid, 1]+sz]
    H = im1_gaussian[avg_x[0]:avg_x[1], avg_y[0]:avg_y[1]]
    h = H.copy()
    h = (h - h.min()) / (h.max() - h.min())
    h[h < h.max() * 0.15] = 0
    num_pixels.append((h > 0).sum())
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.title(f'bead {nid}, pixels {(h>0).sum()}')
    # plt.imshow(H)
    # plt.subplot(1, 2, 2)
    # plt.imshow(h)    
    # plt.show()
    # print(f'bead {nid}')
    # print((h > 0).sum())
num_pixels = np.array(num_pixels)
xy1 = xy[np.where(num_pixels < 400)[0]]  # 400
print(len(xy))
print(len(xy1))

#%%
plt.subplot(1, 2, 1)
plt.imshow(im1_gaussian, vmax=np.percentile(im1_gaussian, 99.9), cmap='gray')
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
for i in range(len(xy)):
    plt.text(xy[i, 1], xy[i, 0], i, fontsize=8, c='red')
plt.title('gaussian img')
plt.subplot(1, 2, 2)
plt.imshow(im1, cmap='gray', vmax=np.percentile(im1, 99.9))
plt.scatter(xy1[:, 1], xy1[:, 0], color='red', alpha=0.3, s=0.5)
plt.title('img with detected neurons')
#plt.imshow(img1, alpha=0.5)
plt.tight_layout()


#%%
img1 = np.zeros(im.shape)
radius = 15
for xx in xy1:
    cv2.circle(img1, [xx[1], xx[0]], radius, 1, -1)
n = len(xy1)
np.save(save_img_folder+'/locs.npy', xy1)

#%% form predictor using camera pixels
pred = map_to_DMD(img1, reg1, reg2)
assert pred[:, 0].max() < 1080 and pred[:, 0].min() > 0
assert pred[:, 1].max() < 1920 and pred[:, 1].min() > 0

#%% create target illumination pattern that focus only on z plane
img = generate_bmp_file(pred, dev=[0, 0])  
img.save(save_dmd_folder + f'/all_beads_radius_{radius}_{date}.bmp')

#%%
core.snap_image()
tagged_image=core.get_tagged_image()
pixels1 = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'],tagged_image.tags['Width']])
filepath = save_img_folder + '\\beads_targeted_1'  + '.tif'
tifffile.imwrite(filepath,pixels1)
plt.figure()
plt.imshow(pixels1, vmax=np.percentile(pixels1, 99), cmap='gray')
plt.savefig(save_img_folder+'/beads_targeted.png')
    
#%% acquire a stack of data with galvo off to extract spatial footprints for reconstruction
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][1]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 20
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}#, 'FrameRate': FrameRate}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)

#%%
save_name = r'origin'
with Acquisition(directory=save_dir, name=save_name, show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=400, time_interval_s=0)
    acq.acquire(events)

#%% generate DMD pattern matrix to simulate fluctuation of fluorescence activity
# the patterns are displayed on DMD 
n = len(xy1)
t = 0.195
fr_orig = 400
tp = int(t*fr_orig) * 5  # 5 frames for 1000/400 = 2.5 ms
n_spikes = 5
t_interval = 0.01
np.random.seed(2023)
radius = 15
intensity_matrix, intensity_all = generate_intensity_matrix(n, t, fr_orig, n_spikes, t_interval)

img_matrix = []
for tt in intensity_matrix.T:
    img1 = np.zeros(im.shape)
    for idx, xx in enumerate(xy1):
        if tt[idx] == 1:
            cv2.circle(img1, [xx[1], xx[0]], radius, 1, -1)
    img_matrix.append(img1)
img_matrix = np.array(img_matrix)

print(intensity_matrix.shape)
print(img_matrix.shape)
print(len(intensity_all))
play(img_matrix, magnification=0.2)

plt.figure()
plt.plot(intensity_all[6])
np.save(save_img_folder+'/intensity.npy', intensity_all)

#%% 
for frame, img1 in enumerate(img_matrix):
    print(frame)
    pred = map_to_DMD(img1, reg1, reg2)
    img = generate_bmp_file(pred)
    img.save(save_dmd_folder + f'/pattern_{frame}.bmp')

with open(save_dmd_folder+f"/load_seq_{date}.txt", "w") as file:  
    file.write("Normal Mode\n")
    for i in range(tp):
        file.write(f"pattern_{i}.bmp,1,105,0,1,0,1,0\n")

#%% acquire streak movies with galvo on 
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][0]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 24.75
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}#, 'FrameRate': FrameRate}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)
print(core.get_property('Andor sCMOS Camera', 'FrameRateLimits'))
print(core.get_property('Andor sCMOS Camera', 'Exposure'))
print(core.get_property('Andor sCMOS Camera', 'Overlap'))    

#%%
save_name = r'test'
with Acquisition(directory=save_dir, name=save_name, show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=120, time_interval_s=0)
    acq.acquire(events)


#%% performance vs cr
cr = [10, 15, 20, 25][3]
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][0]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 1000 / (fr_orig / cr) - 0.15
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}#, 'FrameRate': FrameRate}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)
print(core.get_property('Andor sCMOS Camera', 'FrameRateLimits'))
print(core.get_property('Andor sCMOS Camera', 'Exposure'))
print(core.get_property('Andor sCMOS Camera', 'Overlap'))    

tt = 3
tp = int(tt * fr_orig / cr)
save_name = f'cr_{cr}'

with Acquisition(directory=save_dir, name=save_name, show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=tp, time_interval_s=0)
    acq.acquire(events)
    

#%% performance vs voltage and intensity
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][0]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 24.75
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}#, 'FrameRate': FrameRate}

for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)
    
#%%
cr = 10
tt = 3
levels = 5 # intensity levels
tp = int(tt * fr_orig / cr) * levels
save_name = f'volt_cr_{cr}'

with Acquisition(directory=save_dir, name=save_name,  show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=tp, time_interval_s=0)
    acq.acquire(events)
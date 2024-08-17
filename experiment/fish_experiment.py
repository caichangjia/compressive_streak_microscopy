#!/usr/bin/env python
"""
This file is used to perform zebrafish experiment. It detects neurons, generates DMD pattern for targeted
illumination, and acquires a stack of compressed movie with galvo on 
@author: @caichangjia
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from pycromanager import Core, Acquisition, multi_d_acquisition_events
import os
from skimage import io
from utils import generate_bmp_file, map_to_DMD

#%%
date = '7_20_1'
save_dir = r'C:/Users/nico/Desktop/data/zebrafish_' + date
#save_dir = r'C:/Users/nico/Desktop/data/test_6_12'
fr_orig = 400
fish_id = 1

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

#%% capture widefield movie with red LED light exposure
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][0]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 200-0.25
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)
print(core.get_property('Andor sCMOS Camera', 'FrameRateLimits'))
print(core.get_property('Andor sCMOS Camera', 'Exposure'))
print(core.get_property('Andor sCMOS Camera', 'Overlap'))

#%%    
save_name = r'wf_movie'  # 5Hz, 60s
with Acquisition(directory=save_dir, name=save_name, show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=300, time_interval_s=0)
    acq.acquire(events)
    
#%% compute correlation img
#mov = io.imread('C:/Users/nico/Desktop/data/zebrafish_7_20/wf_movie_1/wf_movie_NDTiffStack.tif')
#mov = io.imread('C:/Users/nico/Desktop/data/zebrafish_7_20_1/wf_movie_1/wf_movie_NDTiffStack.tif')
mov = io.imread('C:/Users/nico/Desktop/data/zebrafish_7_20_1/wf_movie_3/wf_movie_NDTiffStack.tif')
std_img = mov.std(axis=0)
plt.imshow(std_img)

#%% gaussian filter image, gaussian filter is a must for beads detection
im = std_img.copy()
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
#xy = peak_local_max(im_gaussian, min_distance=30, num_peaks=100, exclude_border=False, threshold_abs=0.2)
# img1 = np.zeros((2160, 2560))
# radius = 5
# xy = xy.astype(int)
# for xx in xy:
#     cv2.circle(img1, [xx[1], xx[0]], radius, 1, -1)
# np.save(save_img_folder+'/locs_wf_v3.0.npy', xy)
# print(len(xy))
    
#%%
img1 = np.zeros((2160, 2560))
radius = 5
xy = peak_local_max(im_gaussian, min_distance=10, num_peaks=100, exclude_border=False, threshold_abs=0.2)
centers = xy.copy()
min_dist_p = 55    # the min streak size for 0.04, 0.06, 0.08 is [16, 23, 32] *sqrt(2) + 20
min_dist_v = 20 # neuron diameter estimated as 20 px
num = centers.shape[0]
dis = np.zeros((num, num, 2))  # first dimension parallel to the streak direction, second dimension vertical
for i in range(num):
    for j in range(i+1, num):
        dis[i, j, 1] = np.sqrt(2) / 2 * np.abs(centers[i][0] - centers[j][0] - (centers[i][1] - centers[j][1]))
        dis[i, j, 0] = np.sqrt(2) / 2 * np.abs(centers[i][0] - centers[j][0] + (centers[i][1] - centers[j][1]))

# select neurons that are above min distance away from other neurons
selected = np.array([0])
for i in range(1, num):
    #print(dis[:i ,i][selected])
    # if i == 38:
    #     breakpoint()        
    if sum((dis[:i ,i, 0][selected] < min_dist_p) & (dis[:i ,i, 1][selected] < min_dist_v)) == 0:
        selected = np.append(selected, i)
        print(selected)            
print(len(selected))         
xy = xy[selected]

#%%
plt.subplot(1, 2, 1)
plt.imshow(im, vmax=np.percentile(im, 99.99), cmap='gray')
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
plt.title('raw img')
plt.subplot(1, 2, 2)
plt.imshow(im_gaussian, vmax=np.percentile(im_gaussian, 99.99), cmap='gray')
plt.title('gaussian img')
plt.scatter(xy[:, 1], xy[:, 0], color='red', alpha=0.5, s=0.5)
plt.colorbar()
plt.tight_layout()
#plt.savefig(save_img_folder + '\\detected_peaks_neurons.png')

#%% load calibration result
reg1, reg2 = np.load(save_dir + '\\calib\\regression_result.npy', allow_pickle=True)

#%% form predictor using camera pixels
pred = map_to_DMD(img1, reg1, reg2)
assert pred[:, 0].max() < 1080 and pred[:, 0].min() > 0
assert pred[:, 1].max() < 1920 and pred[:, 1].min() > 0

#%% create targeted illumination pattern on DMD
pred = pred[pred[:, 0] < 1080]
pred = pred[pred[:, 1] < 1920]
img = generate_bmp_file(pred, dev=[0, 0])  

#img.save(save_dmd_folder + f'/test7.bmp')
img.save(save_dmd_folder + f'/neurons_radius_{radius}_wf_v3.0.bmp')

#%% record targeted movie 
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][0]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 200-0.25
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)
print(core.get_property('Andor sCMOS Camera', 'FrameRateLimits'))
print(core.get_property('Andor sCMOS Camera', 'Exposure'))
print(core.get_property('Andor sCMOS Camera', 'Overlap'))

#%%    
save_name = r'tg_movie'  # 5Hz, 60s
with Acquisition(directory=save_dir, name=save_name, show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=300, time_interval_s=0)
    acq.acquire(events)    

#%% acquire streak movie with galvo on 
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][0]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 200-0.25
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}#, 'FrameRate': FrameRate}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)
print(core.get_property('Andor sCMOS Camera', 'FrameRateLimits'))
print(core.get_property('Andor sCMOS Camera', 'Exposure'))
print(core.get_property('Andor sCMOS Camera', 'Overlap'))    

#%%
save_name = r'streak_test'
with Acquisition(directory=save_dir, name=save_name, show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=50, time_interval_s=0)
    acq.acquire(events)

#%% acquire streak movie with galvo on, 20 Hz 
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][0]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 50-0.25
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}#, 'FrameRate': FrameRate}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)
print(core.get_property('Andor sCMOS Camera', 'FrameRateLimits'))
print(core.get_property('Andor sCMOS Camera', 'Exposure'))
print(core.get_property('Andor sCMOS Camera', 'Overlap'))    

#%%
save_name = r'streak_20hz'
with Acquisition(directory=save_dir, name=save_name, show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=1200, time_interval_s=0)
    acq.acquire(events)

#%% acquire streak movie with galvo on, 40 Hz
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][0]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 25-0.25
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}#, 'FrameRate': FrameRate}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)
print(core.get_property('Andor sCMOS Camera', 'FrameRateLimits'))
print(core.get_property('Andor sCMOS Camera', 'Exposure'))
print(core.get_property('Andor sCMOS Camera', 'Overlap'))    

#%%
save_name = r'streak_40hz'
with Acquisition(directory=save_dir, name=save_name, show_display=False) as acq:
    events = multi_d_acquisition_events(num_time_points=2400, time_interval_s=0)
    acq.acquire(events)



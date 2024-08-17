#!/usr/bin/env python
"""
This file is used to map coordinates in camera space to DMD space. Mapping result
is used for targeted illumination
@author: @caichangjia
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from pycromanager import Core
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from sklearn.linear_model import LinearRegression as LR
from scipy.linalg import norm
import tifffile

save_dir = r'C:/Users/nico/Desktop/data/test_8_2/calib'
try:
    os.makedirs(save_dir)
    print('create folder')
except:
    print('folder already created')

#%% setup pycromanager for Andor Zyla camera
core = Core()
TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][1]
AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
ElectronicShutteringMode = ['Global', 'Rolling'][0]
Overlap = ['On', 'Off'][0]
Exposure = 200
properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
              'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
              'Overlap': Overlap}
for key, value in properties.items():
    core.set_property('Andor sCMOS Camera', key, value)

print(core.get_property('Andor sCMOS Camera', 'FrameRateLimits'))
print(core.get_property('Andor sCMOS Camera', 'Exposure'))
print(core.get_property('Andor sCMOS Camera', 'Overlap'))

#%% create a snapshot
core.snap_image()
tagged_image=core.get_tagged_image()
pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'],tagged_image.tags['Width']])
filepath = save_dir + '\\' + 'calibration' + '.tif'
tifffile.imwrite(filepath,pixels)

plt.figure()
plt.imshow(pixels, vmax=np.percentile(pixels, 95))

#%% crop and gaussian filter the image
im = Image.open(save_dir + '\\' + 'calibration' + '.tif')
d = np.array(im)
crop_x = [600, 1400]
crop_y = [600, 1900]
dd = d[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]] # select center region to perform calibration

plt.figure()
plt.imshow(dd)
plt.colorbar()
plt.title('orig img')

ddd = gaussian(dd, sigma=5)
ddd = ddd / ddd.max()

plt.figure()
plt.imshow(ddd)
plt.colorbar()
plt.title('after gaussian filter')

# find local max peaks
xy = peak_local_max(ddd, min_distance=20, threshold_abs=0, exclude_border=False)

plt.figure()
plt.imshow(dd)
plt.colorbar()
plt.scatter(xy[:, 1], xy[:, 0], color='red')

#%% points in the whole FOV
xy[:, 0] = xy[:, 0] + crop_x[0]
xy[:, 1] = xy[:, 1] + crop_y[0]

plt.figure()
plt.imshow(d)
plt.colorbar()
plt.scatter(xy[:, 1], xy[:, 0], color='red')
xy1 = xy.copy()

#%% reshape xy matrix 
dx = [5, 19]
dy = [5, 19]
num_x = dx[1] - dx[0]
num_y = dy[1] - dy[0]
col_rank = np.argsort(xy[:, 0])
xy = xy[col_rank]
xy = xy.reshape((num_y, num_x, 2))
for i in range(num_y):
    xy[i] = xy[i][np.argsort(xy[i][:, 1])]

plt.figure()
plt.imshow(d)
plt.colorbar()
plt.scatter(xy[0:3, 0:3, 1], xy[0:3, 0:3, 0], color='red')  # show points in the topleft region

#%% dmd coordinates, note that the up-side-down of dmd coordinates should match with camera coordinates
dmd_xy = np.array(np.meshgrid(np.arange(dx[0], dx[1]), np.arange(24-dy[0]-1, 24-dy[1]-1, -1))).transpose([1, 2, 0])
dmd_xy = np.flip(dmd_xy, axis=2)
dmd_xy[..., 0] = dmd_xy[..., 0] * 45
dmd_xy[..., 1] = dmd_xy[..., 1] * 80
dmd_xy_flat = dmd_xy.reshape((-1, 2))

plt.figure()
plt.scatter(dmd_xy_flat[:, 1], dmd_xy_flat[:, 0], label='dmd')
xy_flat = xy.reshape((-1, 2))
plt.scatter(xy_flat[:, 1], xy_flat[:, 0], label='camera')
plt.legend()

#%% map camera coordiantes to DMD coordinates by fitting two independent quadratic regression models
xy_mat = np.array([xy_flat[:, 0], xy_flat[:, 1], xy_flat[:, 0] ** 2,  xy_flat[:, 1] ** 2, xy_flat[:, 0]*xy_flat[:, 1]])
reg1 = LR().fit(xy_mat.T, dmd_xy_flat[:, 0])
reg2 = LR().fit(xy_mat.T, dmd_xy_flat[:, 1])

#%% prediction and measure avg deviation, usually ~ 0.5 is good enough
pred = np.array([reg1.predict(xy_mat.T), reg2.predict(xy_mat.T)]).T

plt.figure()
plt.scatter(pred[:, 1], pred[:, 0], color='blue', label='prediction', s=5)
plt.scatter(dmd_xy_flat[:, 1], dmd_xy_flat[:, 0], color='red', label='DMD_pixels', s=5)
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.savefig(save_dir + '\\prediction_DMD_pixels.png')
avg_dev = np.mean(norm(pred - dmd_xy_flat, axis=1))
plt.title(f'calibration with avg deviation {round(avg_dev, 2)}')
print(f'average deviation:{avg_dev}')

#%% save calibration result, it is used in targeted illumination
np.save(save_dir + '\\regression_result' , [reg1, reg2])








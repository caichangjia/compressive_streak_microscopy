#!/usr/bin/env python
"""
This file is used to generate .bmp file for displaying patterns 
including vertical lines, center dots, grid of dots on DMD. 
@author: @caichangjia
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import generate_bmp_file

save_folder = 'C:/Texas Instruments-DLP/DLPC900REF-SW-5.1.0/DLPC900REF-SW-5.1.0/DLPC900REF-GUI/Images and Batch files/Images'

#%% black and white images
img = Image.new( '1', (1920,1080), "black") 
pixels = img.load() 
for i in range(img.size[0]):    
    for j in range(img.size[1]):
        pixels[i,j] = 0 # Set the colour accordingly 0 or 1
img.show()
#img.save(save_folder + '/black.bmp')
#img.save(save_folder + '/white.bmp')

#%% quarter black image, useful to identify how DMD coordinates related to camera coordinates (upside down or other)
img = Image.new( '1', (1920,1080), "black") 
pixels = img.load() 
for i in range(img.size[0]):    
    for j in range(img.size[1]):
        if i < 960 and j < 540:
            pixels[i,j] = 0 
        else:
            pixels[i,j] = 1 
img.show()
#img.save(save_folder + '/black_quarter.bmp')

#%% vertical lines
width = 50
flag = 1

img = Image.new( '1', (1920,1080), "black")
pixels = img.load() 
for i in range(img.size[0]):
    if i % width == 0:
        flag = 1 - flag
    for j in range(img.size[1]):
        pixels[i, j] = flag
img.show()
#img.save(save_folder + '/lines.bmp')

#%% center square
width = 256
img = Image.new( '1', (1920,1080), "black")
pixels = img.load() 
for i in range(img.size[0]): 
    for j in range(img.size[1]):
        if (i >= 1920 // 2 - width // 2) and (i <= 1920 // 2 + width // 2 - 1) and (j >= 1080 // 2 - width // 2) and (j <= 1080 // 2 + width // 2 - 1):
            pixels[i, j] = 0
        else:
            pixels[i, j] = 1
img.show()
#img.save(save_folder + f'/center_width{width}.bmp')

#%% grid, size = 3, offset = 5 is used for calibration DMD and camera
w = 80
h = 45
size = 3     # diameter of the dot
hs = size // 2
offset = 5 # first / last  dots will not take into account

img = Image.new( '1', (1920,1080), "white") 
pixels = img.load() # Create the pixel map
flag = 0
for i in range(w*offset, 1920-offset*w, w):    
    for j in range(h*offset, 1080-offset*h, h):
        print([i, j])
        for k in range(-hs, hs+1, 1):
            for l in range(-hs, hs+1, 1):
                pixels[i+k, j+l] = flag
        
img.show()
#img.save(save_folder + f'/calibration_size_{size}_offset_{offset}.bmp')

#%% grid, used for measuring FWHM
save_folder = 'C:/Texas Instruments-DLP/DLPC900REF-SW-5.1.0/DLPC900REF-SW-5.1.0/DLPC900REF-GUI/Images and Batch files/Images/calibration'
for radius in [1, 3, 5, 10, 15]:
    w = 120
    h = 90
    #size = 9     # diameter of the dot
    #radius = 3
    offset = 1 # first / last  dots will not take into account
    
    pixels = np.zeros((1080, 1920)) # Create the pixel map
    flag = 0
    for i in range(w*(offset+2), 1920-(offset+1)*w, w):    
        for j in range(h*(offset+1), 1080-offset*h, h):
                if radius == 1:
                    pixels[j, i] = 1
                else:
                    cv2.circle(pixels, [i, j], radius, 1, -1)
    pixels = np.round(pixels)
    pixels = np.array(np.where(pixels>0)).T
    img = generate_bmp_file(pixels)
    #img.save(save_folder + f'/fwhm_radius_{radius}_offset_{offset}.bmp')

#%% USAF
p = 'C:/Texas Instruments-DLP/DLPC900REF-SW-5.1.0/DLPC900REF-SW-5.1.0/DLPC900REF-GUI/Images and Batch files/Images/USAF-1951.svg.png'
img = Image.open(p)
#img.show()
img = np.array(img)[..., 1]
img[img >= 128] = 255
img[img < 128] = 0
img1 = img.copy()
plt.imshow(img)

img = Image.new( '1', (1920,1080), "white") 
pixels = img.load() 
for i in range(img1.shape[0]):   
    for j in range(img1.shape[1]):
        try:
            flag = 1 - int(img1[i, img1.shape[1]-j] // 255)
            pixels[i, j] = flag
        except:
            pass
img.show()
#img.save(save_folder + '/USAF.bmp')

#%% lines with different widths
m = np.zeros((1920, 1080))
offset = 150
count = 0
for w in [16, 32, 64, 128]:
    if w > 32:
        m[500:1500, offset+count:offset+count+w] = 1
        count = count + 2*w
    else:
        for j in range(4):
            m[500:1500, offset+count:offset+count+w] = 1
            count = count + 2*w
plt.imshow(m)

img = Image.new( '1', (1920,1080), "black") 
pixels = img.load() 
for i in range(img.size[0]):    
    for j in range(img.size[1]):
        flag = int(m[i, j])
        pixels[i, j] = 1-flag
img.show()
#img.save(save_folder + '/lines_different_width.bmp')

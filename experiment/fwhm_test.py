#!/usr/bin/env python
"""
This file is used to measure optical properties of the system (Supp Fig 1). It measures FWHM of the system
with different DMD circle size.  
@author: @caichangjia
"""
import matplotlib as mpl
import time
import serial
import os
from skimage import io
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from sleep import sleep
import numpy as np
from pycromanager import Core
import tifffile

mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False, 
                     'xtick.major.size':6, 
                     'ytick.major.size':6})

#%%
class Controller:
    '''
    Basic device adaptor for thorlabs MCM3000 and MCM3001 3-axis controllers.
    Not implemented:
    - stop function (not useful?)
    - query motor status (not working? documentation error?)
    Test code runs and seems robust.
    '''
    def __init__(self,
                 which_port,
                 name='MCM3000',
                 stages=3*(None,), # connected e.g. (None, None, 'ZFM2030')
                 reverse=3*(False,), # reverse e.g. (False, False, True)
                 verbose=True,
                 very_verbose=False):
        self.name = name
        self.stages = stages
        self.reverse = reverse        
        self.verbose = verbose
        self.very_verbose = very_verbose
        if self.verbose: print("%s: opening..."%self.name, end='')
        try:
            self.port = serial.Serial(
                port=which_port, baudrate=460800, timeout=5)
        except serial.serialutil.SerialException:
            raise IOError(
                '%s: no connection on port %s'%(self.name, which_port))
        if self.verbose: print(" done.")
        assert type(self.stages) == tuple and type(self.reverse) == tuple
        assert len(self.stages) == 3 and len(self.reverse) == 3
        for element in self.reverse: assert type(element) == bool
        self._encoder_counts= 3*[None]
        self._encoder_counts_tol= 3*[1] # can hang if < 1 count
        self._target_encoder_counts= 3*[None]
        self._um_per_count = 3*[None]
        self._position_limit_um = 3*[None]
        self.position_um = 3*[None]

        supported_stages = { # 'Type': (_um_per_count, +- _position_limit_um, )
                        'ZFM2020':( 0.2116667, 1e3 * 12.7),
                        'ZFM2030':( 0.2116667, 1e3 * 12.7),
                        'MMP-2XY':(0.5, 1e3 * 25.4)}
        self.channels = []
        for channel, stage in enumerate(self.stages):
            if stage is not None:
                assert stage in supported_stages, (
                    '%s: stage \'%s\' not supported'%(self.name, stage))
                self.channels.append(channel)
                self._um_per_count[channel] = supported_stages[stage][0]
                self._position_limit_um[channel] = supported_stages[stage][1]
                self._get_encoder_counts(channel)
        self.channels = tuple(self.channels)
        if self.verbose:
            print("%s: stages:"%self.name, self.stages)
            print("%s: reverse:"%self.name, self.reverse)
            print("%s: um_per_count:"%self.name, self._um_per_count)
            print("%s: position_limit_um:"%self.name, self._position_limit_um)
            print("%s: position_um:"%self.name, self.position_um)

    def _encoder_counts_to_um(self, channel, encoder_counts):
        um = encoder_counts * self._um_per_count[channel]
        if self.reverse[channel]: um = - um + 0 # +0 avoids -0.0
        if self.very_verbose:
            print('%s(ch%s): -> encoder counts %i = %0.2fum'%(
                self.name, channel, encoder_counts, um))
        return um

    def _um_to_encoder_counts(self, channel, um):
        encoder_counts = int(round(um / self._um_per_count[channel]))
        if self.reverse[channel]:
            encoder_counts = - encoder_counts + 0 # +0 avoids -0.0
        if self.very_verbose:
            print('%s(ch%s): -> %0.2fum = encoder counts %i'%(
                self.name, channel, um, encoder_counts))
        return encoder_counts

    def _send(self, cmd, channel, response_bytes=None):
        assert channel in self.channels, (
            '%s: channel \'%s\' is not available'%(self.name, channel))
        if self.very_verbose:
            print('%s(ch%s): sending cmd: %s'%(self.name, channel, cmd))
        self.port.write(cmd)
        if response_bytes is not None:
            response = self.port.read(response_bytes)
        else:
            response = None
        assert self.port.inWaiting() == 0
        if self.very_verbose:
            print('%s(ch%s): -> response: %s'%(self.name, channel, response))
        return response

    def _get_encoder_counts(self, channel):
        if self.very_verbose:
            print('%s(ch%s): getting encoder counts'%(self.name, channel))
        channel_byte = channel.to_bytes(1, byteorder='little')
        cmd = b'\x0a\x04' + channel_byte + b'\x00\x00\x00'
        response = self._send(cmd, channel, response_bytes=12)
        assert response[6:7] == channel_byte # channel = selected
        encoder_counts = int.from_bytes(
            response[-4:], byteorder='little', signed=True)
        if self.very_verbose:
            print('%s(ch%s): -> encoder counts = %i'%(
                self.name, channel, encoder_counts))
        self._encoder_counts[channel] = encoder_counts
        self.position_um[channel] = self._encoder_counts_to_um(
            channel, encoder_counts)
        return encoder_counts

    def _set_encoder_counts_to_zero(self, channel):
        # WARNING: this device adaptor assumes the stage encoder will be set
        # to zero at the centre of it's range for +- stage_position_limit_um checks
        if self.verbose:
            print('%s(ch%s): setting encoder counts to zero'%(
                self.name, channel))
        channel_byte = channel.to_bytes(2, byteorder='little')
        encoder_bytes = (0).to_bytes(4, 'little', signed=True) # set to zero
        cmd = b'\x09\x04\x06\x00\x00\x00' + channel_byte + encoder_bytes
        self._send(cmd, channel)
        while True:
            encoder_counts = self._get_encoder_counts(channel)
            if encoder_counts == 0:
                break
        if self.verbose:
            print('%s(ch%s): -> done'%(self.name, channel))
        return None

    def _move_to_encoder_count(self, channel, encoder_counts, block=True):
        if self._target_encoder_counts[channel] is not None:
            self._finish_move(channel)
        if self.very_verbose:
            print('%s(ch%s): moving to encoder counts = %i'%(
                self.name, channel, encoder_counts))
        self._target_encoder_counts[channel] = encoder_counts
        encoder_bytes = encoder_counts.to_bytes(4, 'little', signed=True)
        channel_bytes = channel.to_bytes(2, byteorder='little')
        cmd = b'\x53\x04\x06\x00\x00\x00' + channel_bytes + encoder_bytes
        self._send(cmd, channel)
        if block:
            self._finish_move(channel)
        return None

    def _finish_move(self, channel, polling_wait_s=0.1):
        if self._target_encoder_counts[channel] is None:
            return
        while True:
            encoder_counts = self._get_encoder_counts(channel)
            if self.verbose: print('.', end='')
            time.sleep(polling_wait_s)
            target = self._target_encoder_counts[channel]
            tolerance = self._encoder_counts_tol[channel]
            if target - tolerance <= encoder_counts <= target + tolerance:
                break
        if self.verbose:
            print('\n%s(ch%s): -> finished move.'%(self.name, channel))
        self._target_encoder_counts[channel] = None
        return None

    def _legalize_move_um(self, channel, move_um, relative):
        if self.verbose:
            print('%s(ch%s): requested move_um = %0.2f (relative=%s)'%(
                self.name, channel, move_um, relative))
        if relative:
            move_um += self.position_um[channel]
        limit_um = self._position_limit_um[channel]
        assert - limit_um <= move_um <= limit_um, (
            '%s: ch%s -> move_um (%0.2f) exceeds position_limit_um (%0.2f)'%(
                self.name, channel, move_um, limit_um))
        move_counts = self._um_to_encoder_counts(channel, move_um)
        legal_move_um = self._encoder_counts_to_um(channel, move_counts)
        if self.verbose:
            print('%s(ch%s): -> legal move_um = %0.2f '%(
                self.name, channel, legal_move_um) +
                  '(%0.2f requested, relative=%s)'%(move_um, relative))
        return legal_move_um

    def move_um(self, channel, move_um, relative=True, block=True):
        legal_move_um = self._legalize_move_um(channel, move_um, relative)
        if self.verbose:
            print('%s(ch%s): moving to position_um = %0.2f'%(
                self.name, channel, legal_move_um))
        encoder_counts = self._um_to_encoder_counts(channel, legal_move_um)
        self._move_to_encoder_count(channel, encoder_counts, block)
        if block:
            self._finish_move(channel)
        return legal_move_um

    def close(self):
        if self.verbose: print("%s: closing..."%self.name, end=' ')
        self.port.close()
        if self.verbose: print("done.")
        return None

def normalize(t):
    return (t - t.min()) / (t.max() - t.min())

#%%
if __name__ == '__main__':
    #%% Setup pycromanager
    core = Core()
    TriggerMode = ['External', 'Internal (Recommended for fast acquisitions)'][1]
    AcquisitionWindow = ['Full Image', '1024x1024', ' 512x512', ' 128x128'][0]
    ElectronicShutteringMode = ['Global', 'Rolling'][0]
    Overlap = ['On', 'Off'][0]
    Exposure = 3000
    properties = {'TriggerMode': TriggerMode, 'AcquisitionWindow': AcquisitionWindow, 
                  'ElectronicShutteringMode': ElectronicShutteringMode, 'Exposure': Exposure, 
                  'Overlap': Overlap}

    for key, value in properties.items():
        core.set_property('Andor sCMOS Camera', key, value)
        
    #%% Setup mcm3000 controller
    channel = 2
    controller = Controller(which_port='COM3',
                            stages=(None, None, 'ZFM2030'),
                            reverse=(False, False, False), 
                            verbose=True,
                            very_verbose=False)
        
    #%% create folder
    #for radius in [1, 3, 5, 10, 15]:
    radius = 15
    save_dir = r'C:/Users/nico/Desktop/data/test_fwhm_7_31'
    save_stack = save_dir + f'/image_stacks_radius_{radius}'
    try:
        os.makedirs(save_stack)
        print('create folder')
    except:
        print('folder already created')

    print('\n# Position attribute = %0.2f'%controller.position_um[channel]) # larger value closer to the slide
    
    print('\n# Home:')
    controller.move_um(channel, -20, relative=False)
    
    #%% recording
    #for d in range(-120, 122, 2):
    for d in range(-20, 26, 1):
        print(f'now processing {d}')
        controller.move_um(channel, d, relative=False)
        sleep(0.1)        
        core.snap_image()
        tagged_image=core.get_tagged_image()
        pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'],tagged_image.tags['Width']])
        filepath = save_stack + f'\\beads_{d+120:03}'  + '.tif'
        tifffile.imwrite(filepath,pixels)

    #%% find centers of "neurons"
    files = sorted(os.listdir(save_stack))
    files = [save_stack + '/' + f for f in files]
    #print(files[])
    pixels = io.imread_collection(files)
    #im = pixels[66][400:1700, 500:2100]
    im = pixels[20][400:1700, 500:2100]

    plt.imshow(im)
    im_gaussian = gaussian(im, sigma=5)
    im_gaussian = im_gaussian / im_gaussian.max()
    plt.figure()
    plt.imshow(im_gaussian)
    plt.colorbar()
    plt.title('after gaussian filter')
    
    # find local max peaks
    #xy = peak_local_max(im_gaussian, min_distance=28, threshold_abs=0.4, num_peaks=1, exclude_border=False)    
    xy = peak_local_max(im_gaussian, min_distance=500, threshold_abs=0.8, num_peaks=10, exclude_border=False)    
    xy = xy[0:1]
    plt.figure()
    plt.imshow(im_gaussian)
    plt.colorbar()
    plt.scatter(xy[:, 1], xy[:, 0], color='red')
    
    #%% process the x and y directions
    w = 50
    width_x = []
    width_y = []
    #bg = im[:100].mean()
    #bg = im[500:600].mean()
    im1 = im# - bg
    tr_x = []
    tr_y = []
    
    for direction in ['x', 'y']:
        for ma in xy:
            x, y = ma
            if direction == 'x':
                tr = im1[x-w:x+w+1, y]
                tr_x.append(tr)
            else:
                tr = im1[x, y-w:y+w+1]
                tr_y.append(tr)
                
            ma = np.argmax(tr)
            maximum = tr[ma]
            half_max = maximum / 2
            idx = np.where(tr>half_max)[0]
            #print(idx)
            width = idx[-1] - idx[0]
            
            if direction == 'x':
                width_x.append(width)
            else:
                width_y.append(width)
    
            # plt.figure()
            # plt.plot(tr)
            # plt.axvspan(idx[0], idx[-1], ymax=maximum, alpha=0.5)
            
    avg_width_x = np.mean(width_x)
    avg_width_y = np.mean(width_y)
    
    print(avg_width_x)
    print(avg_width_y)
    
    tr_x = np.array(tr_x)
    tr_y = np.array(tr_y)
    tr_x_m = np.mean(tr_x, axis=0)
    tr_y_m = np.mean(tr_y, axis=0)
    

    #%% process z directions
    tr_z_all = []
    for xy1 in xy:
        tr_z = []
        for p in pixels:
            p = p[400:1700, 500:2100]
            tr_z.append(p[xy1[0], xy1[1]])
        tr_z_all.append(tr_z)
    tr_z_all = np.array(tr_z_all)
    tr_z_m = tr_z_all.mean(0)
    dic = {'x': tr_x, 'y':tr_y, 'z':tr_z_all}
    
    save_folder = 'C:/Users/nico/Desktop/data/test_fwhm_7_31/result'
    np.save(save_folder + f'/radius_{radius}_result.npy', dic)

    #%% visualization
    for tr in [tr_x_m, tr_y_m, tr_z_m]:
        tr = normalize(tr)
        ma = np.argmax(tr)
        maximum = tr[ma]
        half_max = maximum / 2
        idx = np.where(tr>half_max)[0]
        #print(idx)
        width = idx[-1] - idx[0]
        print(width)
        plt.figure()
        plt.plot(tr)
        plt.axvspan(idx[0], idx[-1], ymax=maximum, alpha=0.5)

    #%% load files
    # files = ['C:/Users/nico/Desktop/data/test_fwhm_7_26/result/radius_1_result_55_center.npy',
    #           'C:/Users/nico/Desktop/data/test_fwhm_7_26/result/radius_3_result_55_center.npy',
    #         'C:/Users/nico/Desktop/data/test_fwhm_7_26/result/radius_5_result_55_center.npy', 
    #         'C:/Users/nico/Desktop/data/test_fwhm_7_26/result/radius_10_result_55_center.npy', 
    #         'C:/Users/nico/Desktop/data/test_fwhm_7_26/result/radius_15_result_55_center.npy']
    
    files = ['C:/Users/nico/Desktop/data/test_fwhm_7_31/result/radius_1_result.npy', 
             'C:/Users/nico/Desktop/data/test_fwhm_7_31/result/radius_3_result.npy',
             'C:/Users/nico/Desktop/data/test_fwhm_7_31/result/radius_5_result.npy', 
             'C:/Users/nico/Desktop/data/test_fwhm_7_31/result/radius_10_result.npy', 
             'C:/Users/nico/Desktop/data/test_fwhm_7_31/result/radius_15_result.npy']

    tx = []
    ty = []
    tz = []
    for f in files:
        fff = np.load(f, allow_pickle=True).item()
        tx.append(fff['x'])
        ty.append(fff['y'])
        tz.append(fff['z'])
        
    #%%
    xm = []
    ym = []
    zm = []
    for xx in tx:
        tr = normalize(xx.mean(0))
        ma = np.argmax(tr)
        maximum = tr[ma]
        half_max = maximum / 2
        idx = np.where(tr>half_max)[0]
        #print(idx)
        width = idx[-1] - idx[0]
        xm.append(width)

    for yy in ty:
        tr = normalize(yy.mean(0))
        ma = np.argmax(tr)
        maximum = tr[ma]
        half_max = maximum / 2
        idx = np.where(tr>half_max)[0]
        #print(idx)
        width = idx[-1] - idx[0]
        ym.append(width)

    for zz in tz:
        tr = normalize(zz.mean(0))
        ma = np.argmax(tr)
        maximum = tr[ma]
        half_max = maximum / 2
        idx = np.where(tr>half_max)[0]
        #print(idx)
        width = idx[-1] - idx[0]
        zm.append(width)

    #%%
    # 1px ~ 0.359 um in x and y directions
    # 1frame = 2um in z direction
    plt.figure(figsize=(20, 5))
    plt.subplot(1,4,1)
    for xx in tx:
        plt.plot(normalize(xx.mean(0)))    
        #plt.plot(xx.mean(0)))
    plt.xticks([50-10/0.359, 50-5/0.359, 50, 50+5/0.359, 50+10/0.359], 
               [-10, -5, 0, 5, 10])
    plt.xlim([50-10/0.359, 50+10/0.359])
    plt.xlabel('Distance to the center (um)')    
    plt.ylabel('Normalized intensity')
    plt.legend(['d = 1', 'd = 7', 'd = 11', 'd = 21', 'd = 31'])
    plt.title('x')
    
    plt.subplot(1,4,2)
    for yy in ty:
        plt.plot(normalize(yy.mean(0)))
    plt.xticks([50-10/0.359, 50-5/0.359, 50, 50+5/0.359, 50+10/0.359], 
               [-10, -5, 0, 5, 10])
    plt.xlim([50-10/0.359, 50+10/0.359])
    plt.xlabel('Distance to the center (um)')    
    plt.ylabel('Normalized intensity')
    plt.title('y')

    plt.subplot(1,4,3)
    for zz in tz:
        zz = zz[:, :]
        plt.plot(normalize(zz.mean(0)))
    plt.xticks([0, 20, 40], [0, 20, 40])
    plt.xlim([0, 40])
#    plt.xlabel('distance to the center (um)')    
    plt.xlabel('Depth in z (um)')    
    plt.ylabel('Normalized intensity')
    plt.title('z')
    
    #
    xt = np.array([1, 7, 11, 21, 31])
    # plt.plot(xt, np.array(xm)*0.359, marker='x', linestyle='dashed', alpha=0.8)    
    # plt.plot(xt, np.array(ym)*0.359, marker='x', linestyle='dotted', alpha=0.8)    
    # plt.plot(xt, np.array(zm)*2, marker='x', linestyle='dashdot', alpha=0.8)    
    mat = np.array([np.array(xm)*0.359, np.array(ym)*0.359, np.array(zm)])
    plt.subplot(1, 4, 4)
    plt.bar(xt-1, mat[0], width=1)
    plt.bar(xt, mat[1], width=1)
    plt.bar(xt+1, mat[2], width=1)
    plt.xticks([1, 7, 11, 21, 31])
    plt.legend(['x', 'y', 'z'])
    plt.xlabel('DMD circle diameter (pixels)')
    plt.ylabel('FWHM (um)')    
    plt.tight_layout()
    
    #plt.savefig('C:/Users/nico/Desktop/data/test_fwhm_7_31/result/optical_properties_one_beadv1.1.pdf')    
    
    #%%
    # 1px ~ 0.359 um in x and y directions
    # 1frame = 2um in z direction
    plt.figure(figsize=(20, 5))
    plt.subplot(1,4,1)
    for xx in tx:
        plt.plot(normalize(xx.mean(0)))    
        #plt.plot(xx.mean(0)))
    plt.xticks([50-15/0.359, 50-10/0.359, 50-5/0.359, 50, 50+5/0.359, 50+10/0.359, 50+15/0.359], 
               [-15, -10, -5, 0, 5, 10, 15])
    plt.xlabel('distance to the center (um)')    
    plt.ylabel('normalized intensity')
    plt.legend(['dimeter 1', 'dimeter 7', 'dimeter 11', 'dimeter 21', 'dimeter 31'])
    plt.title('x')
    
    plt.subplot(1,4,2)
    for yy in ty:
        plt.plot(normalize(yy.mean(0)))
    plt.xticks([50-15/0.359, 50-10/0.359, 50-5/0.359, 50, 50+5/0.359, 50+10/0.359, 50+15/0.359], 
               [-15, -10, -5, 0, 5, 10, 15])
    plt.xlabel('distance to the center (um)')    
    plt.ylabel('normalized intensity')
    plt.title('y')

    plt.subplot(1,4,3)
    for zz in tz:
        zz = zz[:, 20-5:-20-5]
        plt.plot(normalize(zz.mean(0)))
    plt.xticks([0, 20, 40, 60, 80], [0, 20, 40, 60, 80])
    #plt.xlim([10, 110])
#    plt.xlabel('distance to the center (um)')    
    plt.xlabel('depth in z (um)')    
    plt.ylabel('normalized intensity')
    plt.title('z')
    
    #
    xt = np.array([1, 7, 11, 21, 31])
    # plt.plot(xt, np.array(xm)*0.359, marker='x', linestyle='dashed', alpha=0.8)    
    # plt.plot(xt, np.array(ym)*0.359, marker='x', linestyle='dotted', alpha=0.8)    
    # plt.plot(xt, np.array(zm)*2, marker='x', linestyle='dashdot', alpha=0.8)    
    mat = np.array([np.array(xm)*0.359, np.array(ym)*0.359, np.array(zm)*2])
    plt.subplot(1, 4, 4)
    plt.bar(xt-1, mat[0], width=1)
    plt.bar(xt, mat[1], width=1)
    plt.bar(xt+1, mat[2], width=1)
    plt.xticks([1, 7, 11, 21, 31])
    plt.legend(['x', 'y', 'z'])
    plt.xlabel('DMD circle diameter (pixels)')
    plt.ylabel('FWHM (um)')    
    plt.tight_layout()
    
    #plt.savefig('C:/Users/nico/Desktop/data/test_fwhm_7_26/result/optical_properties_v1.1.pdf')

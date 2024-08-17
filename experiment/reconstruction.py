# -*- coding: utf-8 -*-
"""
This file is used to perform reconstruction of high temporal traces for beads and fish experiments.
class Reconstruction is used for beads reconstruction
class ReconstructionFish is used for fish reconstruction
@author: @caichangjia
"""
from concurrent.futures import ThreadPoolExecutor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy.linalg import norm
from sklearn.linear_model import Ridge, Lasso
from scipy.ndimage import label
from sklearn.decomposition import NMF
import time
from utils import hals, normalize

#%%
class Reconstruction:
    def __init__(self, mov, mov_fixed, locs, cr, volt, base_dir, save_dir, method='ridge', 
                 ridge_alpha=0.1, lasso_alpha=0.0001, parallel_processing=True, plot=False):
        """
        Class to reconstruct high temporal resolution traces from streak movie 
        in beads experiments.        
        reconstruct_single_trace method reconstructs the trace of a single ROI.
        reconstruct_traces method reconstructs all traces with parallel processing.

        
        Parameters
        ----------
        mov : float, 3D array
            streak movie, size [t, x, y].
        mov_fixed : flaot, 3D array
            targeted illumination movie to extract masks, size [t, x, y].
        locs : float, 2D array
            centers of ROIs.
        cr : int
            compression ratio.
        volt : float
            galvo input voltage.
        base_dir : str
            base directory for the recording.
        save_dir : str
            saving directory, will save pictures of masks if not None.
        method : str
            reconstruction methods including 'ridge', 'weighted', 'nmf' and 'lasso'. 
            The default is 'ridge'.
        ridge_alpha : float
            ridge regularization strength. The default is 0.1.
        lasso_alpha : float
            lasso regularization strength. The default is 0.0001.
        parallel_processing : bool
            whether to perform parallel processing. The default is True.
        plot : bool
            whether to plot. The default is False.

        Returns
        -------
        all_traces: float, 2D array
            reconstructed traces, size [n_neurons, t].
        C_result: float, 1D array 
            the reconstructed trace.
        A_init: float, 3D array
            updated masks for 'nmf'. size [x, y, cr].    
            None for 'ridge', 'weighted', 'lasso'. 
        masks: flaat, 3D array
            spatial masks for reconstruction. size [cr, x, y].
        """
        self.mov = mov
        self.mov_fixed = mov_fixed
        self.locs = locs
        self.n = locs.shape[0]
        self.mov_mean = mov.mean(0)
        self.cr = cr
        self.volt = volt
        self.method = method
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.parallel_processing = parallel_processing
        self.plot = plot
        self.base_dir = base_dir
        self.save_dir = save_dir
        if self.save_dir is not None:
            try:
                os.makedirs(save_dir)
                print('create folder')
            except:
                print('folder already created')
                
        self.shifts = np.load(base_dir+f'/images/shifts_cr_{cr}_volt_{volt}.npy')
    
    def reconstruct_single_trace(self, nid=None):
        print(f'now processing {nid}')
        ctx_sz = 100 # context size
        avg_sz = 10 # average neuron size
        ctx_x = [self.locs[nid, 0]-ctx_sz, self.locs[nid, 0]+ctx_sz]
        ctx_y = [self.locs[nid, 1]-ctx_sz, self.locs[nid, 1]+ctx_sz]
        avg_x = [self.locs[nid, 0]-avg_sz, self.locs[nid, 0]+avg_sz]
        avg_y = [self.locs[nid, 1]-avg_sz, self.locs[nid, 1]+avg_sz]
        d1 = self.mov[:, ctx_x[0]:ctx_x[1], ctx_y[0]:ctx_y[1]]
        d2 = self.mov_fixed[:, avg_x[0]:avg_x[1], avg_y[0]:avg_y[1]]
        A_init = None

        # extract masks and shifts
        masks = []
        nmf = NMF(n_components=1)
        X = d2.copy().reshape((d2.shape[0], -1))
        W = nmf.fit_transform(X)  
        H = nmf.components_
        H = H.reshape((d2.shape[1], d2.shape[2]))
        h = H.copy()
        h = (h - h.min()) / (h.max() - h.min())
        h[h < h.max() * 0.15] = 0
        mask = np.zeros((ctx_sz*2, ctx_sz*2))
        mask[ctx_sz-avg_sz:ctx_sz+avg_sz, ctx_sz-avg_sz:ctx_sz+avg_sz] = h
        
        masks = []
        for i in range(self.cr):
            affine_matrix = np.array([[1, 0, self.shifts[i, 1]], [0, 1, self.shifts[i, 0]]])  # note the first element to x axis
            masks.append(cv2.warpAffine(mask.copy(),affine_matrix,(ctx_sz*2,ctx_sz*2)))
        masks = np.array(masks)
        masks = masks/norm(masks, axis=(1, 2))[:, np.newaxis, np.newaxis]
        n_masks = masks.shape[0]
        
        if self.plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(H)
            plt.title(f'neuron {nid} mask')
            plt.subplot(1, 2, 2)
            plt.imshow(h)
            plt.title('mask after thresholding')            
            if self.save_dir is not None:
                plt.savefig(self.save_dir+'\\masks_threshold_'+str(nid)+'.png')
            plt.show()                        
            
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.mean(d1, axis=0), cmap='gray', vmax=np.percentile(np.max(d1, axis=0), 95))
            plt.title(f'neuron {nid} mean image')
            plt.subplot(1, 2, 2)
            plt.imshow(np.mean(d1, axis=0), cmap='gray', vmax=np.percentile(np.max(d1, axis=0), 95))
            plt.imshow(masks.sum(0), alpha=0.5)
            plt.title('mean image with masks')            
            if self.save_dir is not None:
                plt.savefig(self.save_dir+'\\masks_'+str(nid)+'.png')
            plt.show()
            time.sleep(random.random())
        
        # extract traces
        Y = d1.copy().transpose([1, 2, 0])
        A = masks.copy()
        A = A / (np.linalg.norm(A, axis=(1, 2), ord='fro')[:, None, None])
        A = A.transpose([1, 2, 0])
        A = A.reshape((-1, A.shape[-1]), order='F')
        A = A.astype('float64')

        if self.method == 'nmf':
            C = np.ones((A.shape[-1], Y.shape[-1]))
            C = C / C.sum(1)[:, None]
            A_init, C = hals(Y=Y, A=A, C=C, b=None, f=None, bSiz=None, maxIter=5, update_shape=True)
            A_init = A_init.reshape((d1.shape[1], d1.shape[2], n_masks), order='F')
        elif self.method == 'weighted':
            Y = Y.reshape((-1, Y.shape[-1]))
            C = A.T@Y
        elif self.method == 'ridge':
            Y = Y.reshape((-1, Y.shape[-1]))
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge.fit(A, Y)
            C = ridge.coef_.T
        elif self.method == 'lasso':
            Y = Y.reshape((-1, Y.shape[-1]))
            lasso = Lasso(alpha=self.lasso_alpha, fit_intercept=False)
            lasso.fit(A, Y)
            C = lasso.coef_.T
            
        # combine traces and normalize
        Cf = C.copy()
        C_result = []
        for i in range(n_masks):
            C_result.append((Cf[i] - Cf[i].mean())/Cf[i].std())
        C_result = np.array(C_result).reshape(-1, order='F')
        C_result = normalize(C_result)
        
        return C_result, A_init, masks

    def reconstruct_traces(self):
        if self.parallel_processing:
            num_threads=10
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(self.reconstruct_single_trace, range(len(self.locs))))
            all_traces = np.array([result[0] for result in results])
        else:
            all_traces = []
            for i in range(len(self.locs)):
                print(f'now processing item {i}')
                C_result, A_init, masks = self.reconstruct_single_trace(nid=i)
                all_traces.append(C_result)
            all_traces = np.array(all_traces)
        return all_traces
    
class ReconstructionFish:
    def __init__(self, mov, mov_fixed, locs, cr, volt, base_dir, save_dir, method='ridge', 
                 ridge_alpha=0.1, lasso_alpha=0.0001, parallel_processing=True, plot=False):
        """
        Class to reconstruct high temporal resolution traces from streak movie 
        in fish experiments.        
        reconstruct_single_trace method reconstructs the trace of a single ROI.
        reconstruct_traces method reconstructs all traces with parallel processing.
        
        Parameters
        ----------
        mov : float, 3D array
            streak movie, size [t, x, y].
        mov_fixed : flaot, 3D array
            targeted illumination movie to extract masks, size [t, x, y].
        locs : float, 2D array
            centers of ROIs.
        cr : int
            compression ratio.
        volt : float
            galvo input voltage.
        base_dir : str
            base directory for the recording
        save_dir : str
            saving directory, will save pictures of masks if not None.
        method : str
            reconstruction methods including 'ridge', 'weighted', 'nmf' and 'lasso'. 
            The default is 'ridge'.
        ridge_alpha : float
            ridge regularization strength. The default is 0.1.
        lasso_alpha : float
            lasso regularization strength. The default is 0.0001.
        parallel_processing : bool
            whether to perform parallel processing. The default is True.
        plot : bool
            whether to plot. The default is False.

        Returns
        -------
        output: list of list
            include n_neuron lists, each list include C_result, C, A, masks, h, C_gt.
        C_result: float, 1D array 
            the reconstructed trace.
        C: float, 2D array
            extracted traces by reconstruction methods, but not combined size [cr, t].
        A: float, 2D array
            spatial masks for reconstruction. Same as 'masks' for 'ridge', 'weighted', 'lasso'.
            Masks are updated for 'nmf'. size [x times y, cr].
        masks: flaat, 3D array
            spatial masks for reconstruction. size [cr, x, y].
        h: float, 2D array
            masks extracted by nmf from targeted movie, size [x, y] .
        C_gt: float, 1D array
            'Ground truth' traces at streak camera sampling rate.            
        """
        
        self.mov = mov
        self.mov_fixed = mov_fixed
        self.locs = locs
        self.n = locs.shape[0]
        self.mov_mean = mov.mean(0)
        self.cr = cr
        self.volt = volt
        self.method = method
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.parallel_processing = parallel_processing
        self.plot = plot
        self.base_dir = base_dir
        self.save_dir = save_dir
        if self.save_dir is not None:
            try:
                os.makedirs(save_dir)
                print('create folder')
            except:
                print('folder already created')
                
        self.shifts = np.load(base_dir+f'/images/shifts_cr_{cr}_volt_{volt}.npy')
    
    def reconstruct_single_trace(self, nid=None):
        print(f'now processing {nid}')
        #%% extract movie, draw context region
        ctx_sz = 100 # context size
        avg_sz = 10 # average neuron size
        ctx_x = [self.locs[nid, 0]-ctx_sz, self.locs[nid, 0]+ctx_sz]
        ctx_y = [self.locs[nid, 1]-ctx_sz, self.locs[nid, 1]+ctx_sz]
        avg_x = [self.locs[nid, 0]-avg_sz, self.locs[nid, 0]+avg_sz]
        avg_y = [self.locs[nid, 1]-avg_sz, self.locs[nid, 1]+avg_sz]
        d1 = self.mov[:, ctx_x[0]:ctx_x[1], ctx_y[0]:ctx_y[1]]
        d2 = self.mov_fixed[:, avg_x[0]:avg_x[1], avg_y[0]:avg_y[1]]
        
        #%% extract masks
        masks = []
        nmf = NMF(n_components=1)
        X = d2.copy().reshape((d2.shape[0], -1))
        W = nmf.fit_transform(X)  
        H = nmf.components_
        H = H.reshape((d2.shape[1], d2.shape[2]))
        h = H.copy()
        h = (h - h.min()) / (h.max() - h.min())
        h[h < h.max() * 0.6] = 0        
        
        # remove extra components when necessary
        labeled_array, num_features = label(h)
        if num_features > 1:
            print('more than one connected componenets')
            connected_size = np.array([len(np.where(labeled_array==k)[0])for k in range(1, num_features+1)])
            connected_max = np.where(connected_size==connected_size.max())[0][0]+1
            temp = np.zeros(h.shape)
            temp[np.where(labeled_array==connected_max)] = 1
            h[temp == 0] = 0
            
        h = h / np.linalg.norm(h)
        mask = np.zeros((ctx_sz*2, ctx_sz*2))
        mask[ctx_sz-avg_sz:ctx_sz+avg_sz, ctx_sz-avg_sz:ctx_sz+avg_sz] = h
        
        #%% compute traces of streak video at camera rate
        masks = []
        for i in range(self.cr):
            affine_matrix = np.array([[1, 0, self.shifts[i, 1]], [0, 1, self.shifts[i, 0]]])  # note the first element to x axis
            masks.append(cv2.warpAffine(mask.copy(),affine_matrix,(ctx_sz*2,ctx_sz*2)))
        masks = np.array(masks)
        masks = masks/norm(masks, axis=(1, 2))[:, np.newaxis, np.newaxis]
        n_masks = masks.shape[0]        
        mask_streak = masks.sum(0)
        mask_streak[mask_streak > 0] = 1
        mask_streak /= mask_streak.sum()
        C_gt = (d1.reshape(d1.shape[0], -1)) @ (mask_streak.reshape(-1))
        
        if self.plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(H)
            plt.title(f'neuron {nid} mask')
            plt.subplot(1, 2, 2)
            plt.imshow(h)
            #plt.imshow(mask, alpha=0.5)
            plt.title('mask after thresholding')
            plt.show()                        
            
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.mean(d1, axis=0), cmap='gray', vmax=np.percentile(np.max(d1, axis=0), 95))
            plt.title(f'neuron {nid} mean image')
            plt.subplot(1, 2, 2)
            plt.imshow(np.mean(d1, axis=0), cmap='gray', vmax=np.percentile(np.max(d1, axis=0), 95))
            plt.imshow(masks.sum(0), alpha=0.5)
            #plt.imshow(mask, alpha=0.5)
            plt.title('mean image with masks')
            # if self.save_dir is not None:
            #     plt.savefig(self.save_dir+'\\masks_'+str(nid)+'.png')
            plt.show()
            time.sleep(random.random())
        
        #%% extract traces
        Y = d1.copy().transpose([1, 2, 0])
        A = masks.copy()
        A = A / (np.linalg.norm(A, axis=(1, 2), ord='fro')[:, None, None])
        A = A.transpose([1, 2, 0])
        A = A.reshape((-1, A.shape[-1]), order='F')
        A = A.astype('float64')

        if self.method == 'nmf':
            C = np.ones((A.shape[-1], Y.shape[-1]))
            C = C / C.sum(1)[:, None]
            A, C = hals(Y=Y, A=A, C=C, b=None, f=None, bSiz=None, maxIter=5, update_shape=True)
            A = A.reshape((d1.shape[1], d1.shape[2], n_masks), order='F')
        elif self.method == 'weighted':
            Y = Y.reshape((-1, Y.shape[-1]))
            C = A.T@Y
        elif self.method == 'ridge':
            Y = Y.reshape((-1, Y.shape[-1]))
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge.fit(A, Y)
            C = ridge.coef_.T
        elif self.method == 'lasso':
            Y = Y.reshape((-1, Y.shape[-1]))
            lasso = Lasso(alpha=self.lasso_alpha, fit_intercept=False)
            lasso.fit(A, Y)
            C = lasso.coef_.T
                    
        #%% reshape traces and normalize
        Cf = C.copy()
        C_result = []
        for i in range(n_masks):
            C_result.append((Cf[i] - Cf[i].mean())/Cf[i].std())
        C_result = np.array(C_result).reshape(-1, order='F')
        C_result = normalize(C_result)
        
        return [C_result, C, A, masks, h, C_gt]
    
    def reconstruct_traces(self):
        if self.parallel_processing:
            num_threads=10
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                output = list(executor.map(self.reconstruct_single_trace, range(len(self.locs))))
        else:
            output = []
            for i in range(len(self.locs)):
                print(f'now processing item {i}')
                out = self.reconstruct_single_trace(nid=i)
                output.append(out)
        return output
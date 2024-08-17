#!/usr/bin/env python
"""
Utility functions for running simulations.
@author: @caichangjia
"""
import cv2
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import scipy.sparse as spr
from scipy import stats    
from scipy.io import savemat
from scipy.optimize import linear_sum_assignment


def save_to_mat(npzFiles, savef):
    for f in npzFiles:
        ff = f.split('/')[-1]
        fm = savef + ff.split('.')[0] + '.mat'
        d = np.load(f)
        savemat(fm, d)
        print('generated ', fm, 'from', f)

def save_movie(fnames, mov):
    if '.hdf5' in fnames:
        with h5py.File(fnames,'w') as h5:
            h5.create_dataset('mov', data=mov)
    else:
        raise Exception('can not load this format')
    return mov

def load_movie(fnames):
    if '.hdf5' in fnames:
        with h5py.File(fnames,'r') as h5:
            mov = np.array(h5['mov'])
    elif '.tif' in fnames:
        mov = imread(fnames)
    else:
        raise Exception('can not load this format')
    return mov

def play(mov, fr=400, backend='opencv', magnification=1, interpolation=cv2.INTER_LINEAR, offset=0, gain=1, q_max=100, q_min=1):
    if q_max < 100:
        maxmov = np.nanpercentile(mov[0:10], q_max)
    else:
        maxmov = np.nanmax(mov)
    if q_min > 0:
        minmov = np.nanpercentile(mov[0:10], q_min)
    else:
        minmov = np.nanmin(mov)
        
    for iddxx, frame in enumerate(mov):
        if backend == 'opencv':
            if magnification != 1:
                frame = cv2.resize(frame, None, fx=magnification, fy=magnification, interpolation=interpolation)
            frame = (offset + frame - minmov) * gain / (maxmov - minmov)
            cv2.imshow('frame', frame)
            if cv2.waitKey(int(1. / fr * 1000)) & 0xFF == ord('q'):
                break            
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    for i in range(10):
        cv2.waitKey(100)

def preplot(image):
    image = np.transpose(image, (1, 2, 0))
    image_color = np.zeros_like(image)
    image_color[:, :, 0] = image[:, :, 2]
    image_color[:, :, 1] = image[:, :, 1]
    image_color[:, :, 2] = image[:, :, 0]
    out_image = np.flipud(np.clip(image_color, 0, 1))
    return out_image[60:, 62:-38, :]


def newplot(image):
    image_color = np.zeros_like(image)
    image_color[:, :, 0] = image[:, :, 2]
    image_color[:, :, 1] = image[:, :, 1]
    image_color[:, :, 2] = image[:, :, 0]
    out_image = np.flipud(np.clip(image_color, 0, 1))
    return out_image[78:, 62:-38, :]

def generate_coded_img(mov, coding, n):
    mov_new = []
    m_temp = np.zeros((128, 128))
    m_img = np.zeros((n, 128, 128))
    for idx, m in enumerate(mov):
        m_coded = m * coding[idx % n]
        if idx % n == 0:
            m_temp = np.zeros((128, 128))
        m_temp = m_temp + m_coded
        if idx % n == n - 1:
            mov_new.append(m_temp)
        if idx < n:
            m_img[idx] = m_coded
    return np.array(mov_new), m_img

def hals(Y, A, C, b, f, bSiz=3, maxIter=5, update_shape=True):
    """ Hierarchical alternating least square method for solving NMF problem

    Y = A*C + b*f

    Args:
       Y:      d1 X d2 [X d3] X T, raw data.
           It will be reshaped to (d1*d2[*d3]) X T in this
           function

       A:      (d1*d2[*d3]) X K, initial value of spatial components

       C:      K X T, initial value of temporal components

       b:      (d1*d2[*d3]) X nb, initial value of background spatial component

       f:      nb X T, initial value of background temporal component

       bSiz:   int or tuple of int
        blur size. A box kernel (bSiz X bSiz [X bSiz]) (if int) or bSiz (if tuple) will
        be convolved with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0.

       maxIter: maximum iteration of iterating HALS.

    Returns:
        the updated A, C, b, f

    Authors:
        Johannes Friedrich, Andrea Giovannucci

    See Also:
        http://proceedings.mlr.press/v39/kimura14.pdf
    """

    # smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    #nb = b.shape[1]  # number of background components
    ind_A = A>1e-10
    ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels

    def HALS4activity(Yr, A, C, iters=2):
        U = A.T.dot(Yr)
        V = A.T.dot(A) + np.finfo(A.dtype).eps
        for _ in range(iters):
            for m in range(len(U)):  # neurons
                C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                               V[m, m], 0, np.inf)
        return C

    def HALS4shape(Yr, A, C, iters=2):
        U = C.dot(Yr.T)
        V = C.dot(C.T) + np.finfo(C.dtype).eps
        for _ in range(iters):
            for m in range(K):  # neurons
                ind_pixels = np.squeeze(ind_A[:, m].toarray())
                A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                           ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                            V[m, m]), 0, np.inf)
        return A

    Ab = A
    Cf = C
    for _ in range(maxIter):
        Cf = HALS4activity(np.reshape(
            Y, (np.prod(dims), T), order='F'), Ab, Cf)
        
        if update_shape:
            Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)

    return Ab, Cf

def mov_interpolation(mov, factor=5, method='nn'):
    mov_new = []
    
    if method == 'linear':
        for f in range(factor // 2):
            mov_new.append(mov[0])
        for m1, m2 in zip(mov[:-1], mov[1:]):
            for f in range(factor):
                mov_new.append(m1 * (factor-f)/factor + m2 * f/factor)
        for f in range(factor // 2 + 1):
            mov_new.append(mov[-1])
        mov_new = np.array(mov_new)    
            
    elif method == 'nn':
        mov_new = np.array([mov] * factor).reshape([-1, mov.shape[1], mov.shape[2]], order='F')
    
    elif method == 'piecewise_linear':
        for i in range(len(mov)):
            if i == 0:
                for f in range(factor):
                    mov_new.append(mov[0])
            elif i == len(mov) - 1:
                for f in range(factor):
                    mov_new.append(mov[-1])
            else:
                for k in range(-(factor//2), factor//2 + 1):
                    mov_new.append(mov[i] + (mov[i + 1] - mov[i - 1]) / (2 * factor -1) * k)
        mov_new = np.array(mov_new)    
    mov_new /= factor
    return mov_new

def generate_streak_mov(mov, cr=10, size=5):
    n = cr * 5                          # number of frames in one streak frame
    speed = 10 / cr * size / 5          # shift of one frame, when cr=10, size=5, move 1 pixel
    padding = 10 * size
    mov_new = []
    nx = mov.shape[1]
    ny = mov.shape[2]
    m_temp = np.zeros((nx + padding, ny))
    m_img = np.zeros((2 * n, nx + padding, ny))

    for idx, m in enumerate(mov):
        m_coded = np.zeros((nx + padding, ny))
        m_coded[:nx] = m 
        nn = idx % n
        affine_matrix = np.array([[1, 0, 0], [0, 1, nn * speed]])  # note the first element to x axis
        m_coded = cv2.warpAffine(m_coded.copy(),affine_matrix,(ny, nx+padding))
        
        if idx % n == 0:
            m_temp = np.zeros((nx + padding, ny))
        m_temp = m_temp + m_coded
        if idx % n == n - 1:
            mov_new.append(m_temp)
        if idx < n:
            m_img[idx] = m_coded
    mov_new = np.array(mov_new)
    
    return mov_new, m_img

def generate_streak_mov_masks(A_init, cr=10, size=5):
    nx, ny, num = A_init.shape
    padding = 10 * size 
    A = []
    for idx in range(cr):
        move = size * (idx % cr) * 10 / cr + size / 2 * 10 /cr
        print(move)
        affine_matrix = np.array([[1, 0, 0], [0, 1, move]])  # note the first element to x axis
        A_temp = np.zeros((num, nx + padding, ny))
        aa = A_init.copy().transpose([2, 0, 1])
        for j in range(num):
            A_temp[j] = cv2.warpAffine(aa[j], affine_matrix, (ny, nx+padding))
        A_temp = np.array(A_temp)
        A.append(A_temp)
    A = np.stack(A).reshape([-1, nx + padding, ny], order='F')
    return A

def add_gaussian_noise(mov, loc, scale):
    return mov + np.random.normal(loc, scale, mov.shape)

# def add_poisson_noise(mov, lam):
#     return mov + np.random.poisson(lam, scale, mov.shape)
    
def normalize(t):
    if len(t.shape) == 1:
        return (t - t.mean()) / t.std()
    else:
        return np.array([(tt - tt.mean()) / tt.std() for tt in t])

def fft2c(x):
    return 1 / np.sqrt(np.prod(x.shape)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(y):
    return np.sqrt(np.prod(y.shape)) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(y)))
        
def SoftThresh(y, t):
    y = np.maximum((np.abs(y) - t), 0) * y / np.abs(y)
    y[np.isnan(y)] = 0
    return y

def compute_spnr(signals, spikes):
    spnr = []
    noise = []
    for idx, sg in enumerate(signals):
        noi = np.std(sg)
        temp = []
        for sp in spikes[idx]:
            if sp > 0:
                temp.append(sg[sp-1:sp+2].max())
        spnr.append(np.mean(temp) / noi)
        noise.append(noi)
    return np.array(spnr), np.array(noise)
        
### VolPy functions
def denoise_spikes(data, window_length, fr=400,  hp_freq=1,  clip=100, threshold_method='adaptive_threshold', 
                   min_spikes=10, pnorm=0.5, threshold=3,  do_plot=True):
    """ Function for finding spikes and the temporal filter given one dimensional signals.
        Use function whitened_matched_filter to denoise spikes. Two thresholding methods can be 
        chosen, simple or 'adaptive thresholding'.

    Args:
        data: 1-d array
            one dimensional signal

        window_length: int
            length of window size for temporal filter

        fr: int
            number of samples per second in the video
            
        hp_freq: float
            high-pass cutoff frequency to filter the signal after computing the trace
            
        clip: int
            maximum number of spikes for producing templates

        threshold_method: str
            adaptive_threshold or simple method for thresholding signals
            adaptive_threshold method threshold based on estimated peak distribution
            simple method threshold based on estimated noise level 
            
        min_spikes: int
            minimal number of spikes to be detected
            
        pnorm: float
            a variable deciding the amount of spikes chosen for adaptive threshold method

        threshold: float
            threshold for spike detection in simple threshold method 
            The real threshold is the value multiply estimated noise level

        do_plot: boolean
            if Ture, will plot trace of signals and spiketimes, peak triggered
            average, histogram of heights
            
    Returns:
        datafilt: 1-d array
            signals after whitened matched filter

        spikes: 1-d array
            record of time of spikes

        t_rec: 1-d array
            recovery of original signals

        templates: 1-d array
            temporal filter which is the peak triggered average

        low_spikes: boolean
            True if number of spikes is smaller than 30
            
        thresh2: float
            real threshold in second round of spike detection 
    """
    # high-pass filter the signal for spike detection
    data = signal_filter(data, hp_freq, fr, order=5)
    data = data - np.median(data)
    pks = data[signal.find_peaks(data, height=None)[0]]

    # first round of spike detection    
    if threshold_method == 'adaptive_threshold':
        thresh, _, _, low_spikes = adaptive_thresh(pks, clip, 0.25, min_spikes)
        locs = signal.find_peaks(data, height=thresh)[0]
    elif threshold_method == 'simple':
        thresh, low_spikes = simple_thresh(data, pks, clip, 3.5, min_spikes)
        locs = signal.find_peaks(data, height=thresh)[0]
    else:
        logging.warning("Error: threshold_method not found")
        raise Exception('Threshold_method not found!')

    # spike template
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]
    PTD = data[(locs[:, np.newaxis] + window)]
    PTA = np.median(PTD, 0)
    PTA = PTA - np.min(PTA)
    templates = PTA

    # whitened matched filtering based on spike times detected in the first round of spike detection
    datafilt = whitened_matched_filter(data, locs, window)    
    datafilt = datafilt - np.median(datafilt)

    # second round of spike detection on the whitened matched filtered trace
    pks2 = datafilt[signal.find_peaks(datafilt, height=None)[0]]
    if threshold_method == 'adaptive_threshold':
        thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=pnorm, min_spikes=min_spikes)  # clip=0 means no clipping
        spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    elif threshold_method == 'simple':
        thresh2, low_spikes = simple_thresh(datafilt, pks2, 0, threshold, min_spikes)
        spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    
    # compute reconstructed signals and adjust shrinkage
    t_rec = np.zeros(datafilt.shape)
    t_rec[spikes] = 1
    t_rec = np.convolve(t_rec, PTA, 'same')   
    factor = np.mean(data[spikes]) / np.mean(datafilt[spikes])
    datafilt = datafilt * factor
    thresh2_normalized = thresh2 * factor
        
    if do_plot:
        plt.figure()
        plt.subplot(211)
        plt.hist(pks, 500)
        plt.axvline(x=thresh, c='r')
        plt.title('raw data')
        plt.subplot(212)
        plt.hist(pks2, 500)
        plt.axvline(x=thresh2, c='r')
        plt.title('after matched filter')
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(np.transpose(PTD), c=[0.5, 0.5, 0.5])
        plt.plot(PTA, c='black', linewidth=2)
        plt.title('Peak-triggered average')
        plt.show()

        plt.figure()
        plt.subplot(211)
        plt.plot(data)
        plt.plot(locs, np.max(datafilt) * 1.1 * np.ones(locs.shape), color='r', marker='o', fillstyle='none',
                 linestyle='none')
        plt.plot(spikes, np.max(datafilt) * 1 * np.ones(spikes.shape), color='g', marker='o', fillstyle='none',
                 linestyle='none')
        plt.subplot(212)
        plt.plot(datafilt)
        plt.plot(locs, np.max(datafilt) * 1.1 * np.ones(locs.shape), color='r', marker='o', fillstyle='none',
                 linestyle='none')
        plt.plot(spikes, np.max(datafilt) * 1 * np.ones(spikes.shape), color='g', marker='o', fillstyle='none',
                 linestyle='none')
        plt.show()

    return datafilt, spikes, t_rec, templates, low_spikes, thresh2_normalized

def adaptive_thresh(pks, clip, pnorm=0.5, min_spikes=10):
    """ Adaptive threshold method for deciding threshold given heights of all peaks.

    Args:
        pks: 1-d array
            height of all peaks

        clip: int
            maximum number of spikes for producing templates

        pnorm: float, between 0 and 1, default is 0.5
            a variable deciding the amount of spikes chosen for adaptive threshold method
            
        min_spikes: int
            minimal number of spikes to be detected

    Returns:
        thresh: float
            threshold for choosing spikes

        falsePosRate: float
            possibility of misclassify noise as real spikes

        detectionRate: float
            possibility of real spikes being detected

        low_spikes: boolean
            true if number of spikes is smaller than minimal value
    """
    # find median of the kernel density estimation of peak heights
    spread = np.array([pks.min(), pks.max()])
    spread = spread + np.diff(spread) * np.array([-0.05, 0.05])
    low_spikes = False
    pts = np.linspace(spread[0], spread[1], 2001)
    kde = stats.gaussian_kde(pks)
    f = kde(pts)    
    xi = pts
    center = np.where(xi > np.median(pks))[0][0]

    fmodel = np.concatenate([f[0:center + 1], np.flipud(f[0:center])])
    if len(fmodel) < len(f):
        fmodel = np.append(fmodel, np.ones(len(f) - len(fmodel)) * min(fmodel))
    else:
        fmodel = fmodel[0:len(f)]

    # adjust the model so it doesn't exceed the data:
    csf = np.cumsum(f) / np.sum(f)
    csmodel = np.cumsum(fmodel) / np.max([np.sum(f), np.sum(fmodel)])
    lastpt = np.where(np.logical_and(csf[0:-1] > csmodel[0:-1] + np.spacing(1), csf[1:] < csmodel[1:]))[0]
    if not lastpt.size:
        lastpt = center
    else:
        lastpt = lastpt[0]
    fmodel[0:lastpt + 1] = f[0:lastpt + 1]
    fmodel[lastpt:] = np.minimum(fmodel[lastpt:], f[lastpt:])

    # find threshold
    csf = np.cumsum(f)
    csmodel = np.cumsum(fmodel)
    csf2 = csf[-1] - csf
    csmodel2 = csmodel[-1] - csmodel
    obj = csf2 ** pnorm - csmodel2 ** pnorm
    maxind = np.argmax(obj)
    thresh = xi[maxind]

    if np.sum(pks > thresh) < min_spikes:
        low_spikes = True
        logging.warning(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
    elif ((np.sum(pks > thresh) > clip) & (clip > 0)):
        logging.warning(f'Selecting top {clip} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))

    ix = np.argmin(np.abs(xi - thresh))
    falsePosRate = csmodel2[ix] / csf2[ix]
    detectionRate = (csf2[ix] - csmodel2[ix]) / np.max(csf2 - csmodel2)
    return thresh, falsePosRate, detectionRate, low_spikes


def simple_thresh(data, pks, clip, threshold=3.5, min_spikes=10):
    """ Simple threshold method for deciding threshold based on estimated noise level.

    Args:
        data: 1-d array
            the input trace
            
        pks: 1-d array
            height of all peaks

        clip: int
            maximum number of spikes for producing templates

        threshold: float
            threshold for spike detection in simple threshold method 
            The real threshold is the value multiply estimated noise level
    
        min_spikes: int
            minimal number of spikes to be detected

    Returns:
        thresh: float
            threshold for choosing spikes

        low_spikes: boolean
            true if number of spikes is smaller than minimal value
    """
    low_spikes = False
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    thresh = threshold * std
    locs = signal.find_peaks(data, height=thresh)[0]
    if len(locs) < min_spikes:
        logging.warning(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
        low_spikes = True
    elif ((len(locs) > clip) & (clip > 0)):
        logging.warning(f'Selecting top {clip} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))    
    return thresh, low_spikes


def whitened_matched_filter(data, locs, window):
    """
    Function for using whitened matched filter to the original signal for better
    SNR. Use welch method to approximate the spectral density of the signal.
    Rescale the signal in frequency domain. After scaling, convolve the signal with
    peak-triggered-average to make spikes more prominent.
    
    Args:
        data: 1-d array
            input signal

        locs: 1-d array
            spike times

        window: 1-d array
            window with size of temporal filter

    Returns:
        datafilt: 1-d array
            signal processed after whitened matched filter
    
    """
    N = np.ceil(np.log2(len(data)))
    censor = np.zeros(len(data))
    censor[locs] = 1
    censor = np.int16(np.convolve(censor.flatten(), np.ones([1, len(window)]).flatten(), 'same'))
    censor = (censor < 0.5)
    noise = data[censor]

    _, pxx = signal.welch(noise, fs=2 * np.pi, window=signal.get_window('hamming', 1000), nfft=2 ** N, detrend=False,
                          nperseg=1000)
    Nf2 = np.concatenate([pxx, np.flipud(pxx[1:-1])])
    scaling_vector = 1 / np.sqrt(Nf2)

    cc = np.pad(data.copy(),(0,int(2**N-len(data))),'constant')    
    dd = (cv2.dft(cc,flags=cv2.DFT_SCALE+cv2.DFT_COMPLEX_OUTPUT)[:,0,:]*scaling_vector[:,np.newaxis])[:,np.newaxis,:]
    dataScaled = cv2.idft(dd)[:,0,0]
    PTDscaled = dataScaled[(locs[:, np.newaxis] + window)]
    PTAscaled = np.mean(PTDscaled, 0)
    datafilt = np.convolve(dataScaled, np.flipud(PTAscaled), 'same')
    datafilt = datafilt[:len(data)]
    return datafilt

from scipy import signal
def signal_filter(sg, freq, fr, order=3, mode='high'):
    """
    Function for high/low passing the signal with butterworth filter
    
    Args:
        sg: 1-d array
            input signal
            
        freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
    Returns:
        sg: 1-d array
            signal after filtering            
    """
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg

# match spikes 
def compute_distances(s1, s2, max_dist):
    """
    Define a distance matrix of spikes.
    Distances greater than maximum distance are assigned one.

    Parameters
    ----------
    s1,s2 : ndarray
        Spikes time of two methods
    max_dist : int
        Maximum distance allowed between two matched spikes

    Returns
    -------
    D : ndarray
        Distance matrix between two spikes
    """
    D = np.ones((len(s1), len(s2)))
    for i in range(len(s1)):
        for j in range(len(s2)):
            if np.abs(s1[i] - s2[j]) > max_dist:
                D[i, j] = 1
            else:
                # 1.01 is to avoid two pairs of matches 'cross' each other
                D[i, j] = (np.abs(s1[i] - s2[j]))/5/max_dist ** 1.01 
    return D

def match_spikes_linear_sum(D):
    """
    Find matches among spikes by solving linear sum assigment problem.
    Delete matches where their distances are greater than the maximum distance.
    Parameters
    ----------
    D : ndarray
        Distance matrix between two spikes
        
    Returns
    -------
    idx1, idx2 : ndarray
        Matched spikes indexes

    """
    idx1, idx2 = linear_sum_assignment(D)
    del_list = []
    for i in range(len(idx1)):
        if D[idx1[i], idx2[i]] == 1:
            del_list.append(i)
    idx1 = np.delete(idx1, del_list)
    idx2 = np.delete(idx2, del_list)
    return idx1, idx2

def match_spikes_greedy(s1, s2, max_dist):
    """
    Match spikes using the greedy algorithm. Spikes greater than the maximum distance
    are never matched.
    Parameters
    ----------
    s1,s2 : ndarray
        Spike time of two methods
    max_dist : int
        Maximum distance allowed between two matched spikes

    Returns
    -------
    idx1, idx2 : ndarray
        Matched spikes indexes with respect to s1 and s2

    """
    l1 = list(s1.copy())
    l2 = list(s2.copy())
    idx1 = []
    idx2 = []
    temp1 = 0
    temp2 = 0
    while len(l1) * len(l2) > 0:
        if np.abs(l1[0] - l2[0]) <= max_dist:
            idx1.append(temp1)
            idx2.append(temp2)
            temp1 += 1
            temp2 += 1
            del l1[0]
            del l2[0]
        elif l1[0] < l2[0]:
            temp1 += 1
            del l1[0]
        elif l1[0] > l2[0]:
            temp2 += 1
            del l2[0]
    return idx1, idx2

def compute_F1(s1, s2, idx1, idx2):
    """
    Compute F1 scores, precision and recall.

    Parameters
    ----------
    s1,s2 : ndarray
        Spike time of two methods. Note we assume s1 as ground truth spikes.
    
    idx1, idx2 : ndarray
        Matched spikes indexes with respect to s1 and s2

    Returns
    -------
    F1 : float
        Measures of how well spikes are matched with ground truth spikes. 
        The higher F1 score, the better.
        F1 = 2 * (precision * recall) / (precision + recall)
    precision, recall : float
        Precision and recall rate of spikes matching.
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    """
    TP = len(idx1)
    FP = len(s2) - TP
    FN = len(s1) - TP
    
    if len(s1) == 0:
        F1 = np.nan
        precision = np.nan
        recall = np.nan
    else:
        try:    
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
        recall = TP / (TP + FN)
        try:
            F1 = 2 * (precision * recall) / (precision + recall) 
        except ZeroDivisionError:
            F1 = 0
            
    return F1, precision, recall
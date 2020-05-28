import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF

def flatten_im(im):
    '''
    Flattens 3D data cube into 2D array for NMF processing.

    Parameters
    ----------
    im : numpy array
        3D image array

    Returns
    ----------
    im2d : numpy array
        2D image array
    '''
    new_0 = im.shape[0]**2
    new_1 = im.shape[2]

    im2d = np.zeros((new_0, new_1))

    for i in range(new_1):
        im2d[:,i] = im[:,:,i].flatten()

    return im2d


def run_nmf(im, comps, solver = 'cd', tol = 1e-5, max_iter=int(1e6), l1_ratio = 0.2, alpha = 1e-3, verbose = False):
    '''
    Performs NMF analysis on a give image.

    Processing wrapper for NMF on 3D data cubes for hyperspectral images with a given number of components.

    Parameters
    ----------
    im : numpy array
        3D data cube of hyperspectral image
    comps : integer
        Number of components that NMF will try to identify
    solver : string (optional)
        Either 'cd' or 'mu' for coordinate descent or multiplicative update; default 'cd'
    tol : float (optional)
        Tolerance factor; adjust to smaller if analysis requires more precision; default 1e-5
    max_iter : integer (optional)
        Number of iterations NMF algorithm will attempt to separate components; adjust as necessary for attempts and/or time constraints; default int(1e6)
    l1_ratio : float (optional)
        Algorithm step size factor; adjust to smaller or larger to change fitting/overfitting; default 0.2
    alpha : float (optional)
        Algorithm step size factor; default 1e-3
    verbose : Boolean (optional)
        Determines whether or not user views epoch calculations; default False

    Returns
    ----------
    H : numpy array (size comps x im.shape[2])
        Model components (e.g. spectra of components)
    W : numpy array (size comps x im.shape[0]^2)
        Model coefficient matrices (flattened)
    '''
    im2d = flatten_im(im)

    model = NMF(n_components = comps, solver = solver, tol = tol, max_iter=max_iter, l1_ratio = l1_ratio, alpha = alpha, random_state = 0, verbose = verbose)

    W = model.fit_transform(im2d + abs(np.min(im2d)))

    H = model.components_

    return W, H


def check_comps(comps, truths):
    '''
    Normalizes and matches components with ground truth spectra for comparison.

    Minimizes root mean squared error values for NMF determined components and provided ground truth spectra.

    Parameters
    ----------
    comps : numpy array
        Output of NMF analysis; H matrix
    truths : dictionary
        Dictionary of ground truth labels and their corresponding spectra

    Returns
    ----------
    matches : dictionary
        Dictionary of matches between components and ground truths with minimized RMSE. Ground truth spectra are keys.'''

    matches = {}

    # normalize the components
    for i in range(comps.shape[0]):
        comps[i,:] = np.interp(comps[i,:], (comps[i,:].min(), comps[i,:].max()), (0,1))

    # initialize matches dictionary
    for key in truths:
        matches[key] = {'truth':truths[key],
                        'NMF':None,
                        'RMSE':None,
                        'Index':None}

    # initialize empty rmse_vals comparison dictionary
    rmse_vals = {}

    # what's in the rmse_vals dictionary
    for key in truths:
        rmse_vals[key] = {} # for each ground truth spectra, initialize that as a dictionary, too
        for i in range(comps.shape[0]):
            # within the key (ground truths) set each index (# of comps) to the RMSE
            rmse_vals[key][i] = mean_squared_error(truths[key], comps[i,:], squared = False)

    for i in range(comps.shape[0]): # check each component
        val = 1000000000
        min_key = None
        for key in truths: # for each ground truth key
            if val > rmse_vals[key][i]:
                val = rmse_vals[key][i]
                min_key = key
            else:
                pass
        matches[min_key]['NMF'] = comps[i,:]
        matches[min_key]['RMSE'] = val
        matches[min_key]['Index'] = i

    return matches


def norm_W(W, index):
    '''
    Normalize the coefficient matrices (output of NMF analysis).

    Parameters
    ----------
    W : numpy array
        2D numpy array of shape size^2 x # components

    index : integer
        Coefficient matrix to be normalized to

    Returns
    ----------
    threshold_mtx : numpy array
        Normalized coefficient matrix of the given index.
    '''
    size = int(np.sqrt(W.shape[0]))

    W_mat = W.reshape((size, size, W.shape[1]))

    threshold_mtx = np.zeros((size, size))

    sum_mtx = np.zeros((size, size))

    for i in range(W_mat.shape[2]):
        sum_mtx += W_mat[:,:,i]

    for i in range(threshold_mtx.shape[0]):
        for j in range(threshold_mtx.shape[1]):
            threshold_mtx[i,j] = W_mat[i,j,index]/sum_mtx[i,j]

    return threshold_mtx


def mask_im(t_mtx, threshold):
    '''
    Binarize a threshold matrix given a threshold.

    Designed to identify binary regions of given composition relative to a known index.

    Parameters
    ----------
    t_mtx : 2D numpy array
        Threshold matrix (output of norm_W function)
    threshold : float (between 0 and 1)
        Threshold value corresponding to percent present in question

    Returns
    ----------
    mask : 2D numpy array
        Binarized image from threshold
    '''
    mask = np.zeros_like(t_mtx)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if t_mtx[i,j] >= threshold:
                mask[i,j] = 1
            else:
                mask[i,j] = 0

    return mask

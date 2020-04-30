import numpy as np

def avg_spectra(im):
    
    """
    avg_spectra(im)

    Returns numpy.ndarray of mean spectrum, averaged over the image pixels

    Parameters
    ----------
    im : image passed as numpy array

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.

    """
    
    slice_list = np.dsplit(im, im.shape[2])
    means_list = [np.mean(i) for i in slice_list]
    
    return np.array(means_list)

def area_image(im):
    """
    area_spectra(im)

    Returns numpy.ndarray of image with each pixel having the area under its spectrum

    Parameters
    ----------
    im : image passed as numpy array

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.

    """
    
    x = 750 + np.linspace(0,im.shape[2]-1,im.shape[2])
    return np.trapz(im, x, axis=2)
    
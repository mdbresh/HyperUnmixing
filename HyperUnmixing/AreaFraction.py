import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb

def AreaFraction(im, norm_im, image_size):
    '''
    Takes a test image, the NMF coefficients normalized image, and the image size and creates 
    a threshold mask of the image, labels regions of interest, and computes the region
    properties. It then computes the fraction of the total image that is a certain 
    percentage (or higher) of a polymer component.
    
    Parameters
    -----------------   
    im : ndarray.
         Original image to process.
        
    norm_im : ndarray.
              Image of NMF coefficients, normalized to be less than 1.
              
    image_size : int.
                 Size of the total image. 
   
    Returns
    ----------------
    regions : dict.
              Dictionary of regions of interest and their computed properties.

    mask : ndarray.
           Multidimensional array with each binarized image.
          
    area_frac : list.
                List of computed area fractions of DPPDTT. '''
    
    ## Set up an array of threshold values
    percents = np.round(np.arange(0.5, 1.0, 0.05),2)
    
    ## Create a multidimensional array to store each of the thresholded images. 
    ## In this case, there are ten different tested percentages
    mask = np.zeros((norm_im.shape[0], norm_im.shape[1], 10))

    for h in range(mask.shape[2]):
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if norm_im[i][j] >= percents[h]:
                    mask[i][j][h] = 1
                else:
                    mask[i][j][h] = 0
       
            
    ## Loop through the masks and compute labels and region properties
    regions = {}
    props = ('area', 'major_axis_length', 'minor_axis_length', 'mean_intensity')
    
    for i in range(mask.shape[2]):
        labels = label(mask[:,:,i])
        regions[i] = pd.DataFrame(regionprops_table(labels, im, props))
    
    ## Compute the area fractions
    area_frac = []
    for i in range(len(regions.keys())):
        area_frac.append(regions[i]['area'].values / image_size**2)
        
    return regions, mask, area_frac
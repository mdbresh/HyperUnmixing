import numpy as np
import pandas as pd

from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb

def Wav_2_Im(im, wn):
	'''
	Input a 3-D datacube and outputs a normalized slice at one wavenumber.

	Parameters
	----------
	im : array-like image.
		 Input data.

	wn : integer.
		 Integer index value.

	Returns
	----------
	slice : ndarray.
			An image the same size as the input, but with one slice in wavenumber space.
	'''

	normalized = [] # storage for each normalized slice
	img_norm = np.empty(im.shape, dtype=np.float32)

	for i in np.linspace(0, im.shape[2]-1, im.shape[2]-1).astype(np.int):
		image = im[:,:,i]

		normalized.append((image - np.min(image))/(np.amax(image) - np.min(image)))

	for i in np.linspace(0, im.shape[2]-1, im.shape[2]-1).astype(np.int):
		img_norm[:,:,i] = normalized[i-1]

	im_slice = img_norm[:,:,wn-750]

	return im_slice


def AreaFraction(im, norm_im, image_size):
	'''
	Input test image, normalized NMF coefficients image, and image size. 
	Outputs a dictionary of computed properties for regions of interest, 
	a multidimensional array containing threshold masks, and a list of
	computed area fractions for the areas of interest in each threshold mask.

	Parameters
	----------

	im : array-like image.
		 Image slice to measure.

	norm_im : multidimensional array-like image
			  Image of normalized NMF coefficients.

	image_size : integer.
				 Size of the image.

	Returns
	---------

	regions : dict.
			  Dictionary of regions of interest and their computed properties.

	mask : multidimensional array-like image.
		   Multidimensional array with each threshold mask image.

	area_frac : list.
				List of computed area fractions of DPPDTT.
	'''

	# Set up threshold masks
	percents = np.round(np.arange(0.5, 1.0, 0.05),2) # array of thresholds
	mask = np.zeros((norm_im.shape[0], norm_im.shape[1], 10)) # ten tested thresholds

	for h in range(mask.shape[2]):
		for i in range(mask.shape[0]):
			for j in range(mask.shape[1]):
				if norm_im[i][j] >= percents[h]:
					mask[i][j][h] = 1
				else:
					mask[i][j][h] = 0

	# Compute region properties of labeled images
	regions = {}
	props = ('area', 'major_axis_length', 'minor_axis_length', 'mean_intensity')

	for i in range(mask.shape[2]):
		labels = label(mask[:,:,i])
		regions[i] = pd.DataFrame(regionprops_table(labels, im, props))

	# Compute the area fractions
	area_frac = []
	for i in range(len(regions.keys())):
		area_frac.append(regions[i]['area'].values / image_size**2)

	return regions, mask, area_frac









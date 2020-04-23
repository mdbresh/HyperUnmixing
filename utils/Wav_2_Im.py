def Wav_2_Im(im, wn):

	""" 

	Input a 3-dimensional 'image cube' and outputs a normalized slice at one wavenumber.

	Parameters

	----------

	im : array_like image
		 Input data.

	wn : integer
		 Integer index value.


	Returns

	----------

	slice : ndarray
			An image the same size as the input, but only 1 slice in the 3rd dimension.


	Notes

	----------

	The input image is normalized by individually normalizing each slice in the 3rd dimension. 
	The normalization scheme is (im_slice - minimum of im_slice) / (maximum of im_slice - minimum of im_slice).
	There may be a different way to approach the normalization.


	Examples

	----------

	>>> image = np.load('image1.npy')
	>>> image.shape
		(256, 256, 1128)
	>>> image_1490 = Wav_2_Im(image, 1490)
	>>> image_1490.shape
		(256, 256)
	>>> plt.imshow(image_1490)
	>>> plt.colorbar(label = "intensity")
	>>> plt.show()

	You can change the wavenumber to change the image that is visualized.

	"""

	## Set up storage for each normalized slice
    normalized = []
    
    ## Set up storage for the entire normalized image
    img_norm = np.empty(image.shape, dtype=np.float32)
    
    ## Loop through each slice in the image and normalize it by: (slice - slice min)/(slice max - slice min)
    for i in np.linspace(0, image.shape[2]-1, image.shape[2]-1).astype(np.int):
    
        ## pull out one slice
        im = image[:,:,i]
    
        ## normalize the slice
        normalized.append((im - np.min(im))/(np.amax(im) - np.min(im)))  

    ## Loop through each slice in the storage array and replace it with the normalized slice
    for i in np.linspace(0, image.shape[2]-1, image.shape[2]-1).astype(np.int):
    
        img_norm[:,:,i] = normalized[i-1]
    
    ## Pull out the normalized image at the wavenumber of interest
    slice = img_norm[:,:,wn-750]
  
    
    return slice




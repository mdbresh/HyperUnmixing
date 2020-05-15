import numpy as np

def normalization(image,xrange=1):
    '''
    normalization function for image
    ------
    Parameters

    Image : array

        hyperspectral images with 3D dataset

    xrange: output range of the spectra at each point, optional

        set to 1

    -------
    Returns

    output : array of float

       normalized hyperspectral image with 3D dataset

    ------
    Examples

    >>>from HyperUnmixing import normalization

    >>>image = np.load('image1.npy')
    >>>new_image=normalization(image)
    >>>plt.plot(image[25,25,:], label='Orginal')
    >>>plt.plot(new_image[25,25,:], label= 'Normalized')
    >>>plt.legend()

    '''

    x,y,z=image.shape
    new_images=np.zeros((x,y,z))
    count =0

    for i in range (0,x):
        for d in range(0,y):
            count +=1
            new_images[i,d,:]=(image[i,d,:]-np.amin(image[i,d,:]))*xrange/(np.amax(image[i,d,:])-np.amin(image[i,d,:]))

    return new_images

from random import randint

def sample(image, size):
    """
    creates a subsample of a 3-dimensional image

    Given a 3D image (x, y, z), this function creates a square subsample of a given size (size, size, z). The third dimension is preserved. Depends on Python random `randint` function.

    Parameters
    ------------
    image : Numpy array of 3 dimensions
        The first two dimensions of the array will be sampled/cropped; the third dimension will be preserved
    size : integer
        The size of the sample to be created

    Returns:
    ------------
    sample : Numpy array of 3 dimensions
        Subsample of original image of given size with intact third dimension
    """
    # define x and y dimensions where valid random uppler left corner may be initiated
    valid_range_x = image.shape[0] - size
    valid_range_y = image.shape[1] - size

    # define x and y coordinates of upper left corner of sampled image
    start_x = randint(0, valid_range_x)
    start_y = randint(0, valid_range_y)

    # grab sample out of original image
    sample = image[start_x:start_x + size,
                   start_y:start_y + size, :]

    return sample

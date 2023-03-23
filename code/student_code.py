import numpy as np


def my_imfilter(image, filter):
    """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###
    # Get the size of the input image and the filter
    im_height, im_width, im_channels = image.shape
    f_height, f_width = filter.shape

    # Initialize the output image
    filtered_image = np.zeros_like(image)

    # Compute the padding required based on the filter size
    pad_height = f_height // 2
    pad_width = f_width // 2

    # Pad the input image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

    # Iterate over each pixel in the input image
    for i in range(im_height):
        for j in range(im_width):
            for k in range(im_channels):
                # Compute the neighborhood of the pixel based on the filter size
                neighborhood = padded_image[i:i + f_height, j:j + f_width, k]

                # Multiply the filter with the pixel values in the neighborhood,
                # sum the results, and store it in the corresponding pixel in the output image.
                filtered_image[i, j, k] = np.sum(neighborhood * filter)



    ### END OF STUDENT CODE ####
    ############################

    return filtered_image


def create_hybrid_image(image1, image2, filter):
    """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    ############################
    ### TODO: YOUR CODE HERE ###


    # Apply Gaussian filter to image1
    low_frequencies = my_imfilter(image1, filter)

    # Apply Laplacian filter to image2 to obtain high-frequency content
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    high_frequencies = image2 - my_imfilter(image2, laplacian_filter)

    # Create hybrid image by adding low and high frequency components
    hybrid_image = np.clip(low_frequencies + high_frequencies, 0, 1)
    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image

# Christerpher Hunter
# HW_06 #1


from typing import Any
from numpy import array, fliplr, flipud, ndarray, zeros
from colorama import Fore as F


R = F.RESET


def main():

    # Image
    #
    f = array([[0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0]])

    """f =  array([[-1.09256581, -0.60177391,  1.24215822],
       [-0.44548082, -1.13145029, -0.54335643],
       [ 0.10972916,  1.96715443,  0.49558543]])"""


    # Kernel
    #
    g = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    output = convolution_2d_image_array(f, g,)
    print(output)


def convolution_2d_image_array(f, g, padding=1, stride=1) -> ndarray:

    g = flipud(fliplr(g))

    # Image shape
    #
    f_x_dim_shape = f.shape[0]
    f_y_dim_shape = f.shape[1]

    # Kernel shape
    #
    g_x_dim_shape = g.shape[0]
    g_y_dim_shape = g.shape[1]

    # Size of the output image
    #
    x_out = int(((f_x_dim_shape - g_x_dim_shape + 2 * padding) / stride) + 1)
    y_out = int(((f_y_dim_shape - g_y_dim_shape + 2 * padding) / stride) + 1)
    output = zeros((x_out, y_out))

    W = array([[-1.09256581, -0.60177391,  1.24215822],
       [-0.44548082, -1.13145029, -0.54335643],
       [ 0.10972916,  1.96715443,  0.49558543]])


    # Apply equal padding to all sides
    #
    if padding != 0:
        padded_image = zeros((f_x_dim_shape + padding * 2,
                              f_y_dim_shape + padding * 2))
        padded_image[int(padding):int(-1 * padding),
                     int(padding):int(-1 * padding)] = f
        # print(padded_image)
    else:
        padded_image = f

    for y in range(f_y_dim_shape):
        # Quit if image is smaller than kernel
        #
        if y > f.shape[1] - g_y_dim_shape:
            break
        # Only do the convolution if y has reduced by the specified strides
        #
        if y % stride == 0:
            for x in range(f.shape[0]):
                # Proceed to the next row
                #
                if x > f.shape[0] - g_x_dim_shape:
                    break
                try:
                    # Only do the convolution if x has reduced by the specified strides
                    #
                    if x % stride == 0:
                        output[x, y] = (
                            g * padded_image[x: x + g_x_dim_shape, y: g_y_dim_shape]).sum()
                except:
                    break

    return output


if __name__ == "__main__":
    main()

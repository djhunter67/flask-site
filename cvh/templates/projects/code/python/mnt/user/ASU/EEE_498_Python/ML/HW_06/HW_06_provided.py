import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
Image_dim = 7
stride = 2
Ydim = 3
Wdim = 3
##
# target
##
target = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [
                  0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]], float)
##
# training data
##
# 0
image7by7 = np.zeros([Image_dim, Image_dim, 9], float)
image7by7[1, :, 0] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[2, :, 0] = np.array([0, 0, 1, 1, 0, 0, 0])
image7by7[3, :, 0] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[4, :, 0] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[5, :, 0] = np.array([0, 0, 1, 1, 1, 0, 0])
# 1
image7by7[1, :, 1] = np.array([0, 0, 1, 1, 0, 0, 0])
image7by7[2, :, 1] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[3, :, 1] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[4, :, 1] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[5, :, 1] = np.array([0, 0, 0, 1, 0, 0, 0])
# 2
image7by7[1, :, 2] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[2, :, 2] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[3, :, 2] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[4, :, 2] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[5, :, 2] = np.array([0, 1, 1, 1, 1, 1, 0])
# 3
image7by7[1, :, 3] = np.array([0, 1, 1, 1, 1, 1, 0])
image7by7[2, :, 3] = np.array([0, 1, 0, 0, 1, 0, 0])
image7by7[3, :, 3] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[4, :, 3] = np.array([0, 1, 1, 0, 0, 0, 0])
image7by7[5, :, 3] = np.array([0, 1, 1, 1, 1, 1, 0])
# 4
image7by7[1, :, 4] = np.array([0, 0, 1, 1, 1, 0, 0])
image7by7[2, :, 4] = np.array([0, 0, 0, 0, 1, 0, 0])
image7by7[3, :, 4] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[4, :, 4] = np.array([0, 0, 1, 0, 0, 0, 0])
image7by7[5, :, 4] = np.array([0, 0, 1, 1, 1, 0, 0])
# 5
image7by7[1, :, 5] = np.array([0, 0, 1, 1, 0, 0, 0])
image7by7[2, :, 5] = np.array([0, 1, 0, 0, 1, 0, 0])
image7by7[3, :, 5] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7[4, :, 5] = np.array([0, 0, 1, 0, 0, 0, 0])
image7by7[5, :, 5] = np.array([0, 1, 1, 1, 1, 1, 0])
# 6
image7by7[1, :, 6] = np.array([0, 1, 1, 1, 1, 1, 0])
image7by7[2, :, 6] = np.array([0, 0, 0, 0, 0, 1, 0])
image7by7[3, :, 6] = np.array([0, 0, 0, 0, 1, 1, 0])
image7by7[4, :, 6] = np.array([0, 0, 0, 0, 0, 1, 0])
image7by7[5, :, 6] = np.array([0, 1, 1, 1, 1, 1, 0])
# 7
image7by7[1, :, 7] = np.array([0, 0, 1, 1, 1, 0, 0])
image7by7[2, :, 7] = np.array([0, 1, 0, 0, 0, 1, 0])
image7by7[3, :, 7] = np.array([0, 0, 0, 1, 1, 0, 0])
image7by7[4, :, 7] = np.array([0, 1, 0, 0, 0, 1, 0])
image7by7[5, :, 7] = np.array([0, 0, 1, 1, 1, 0, 0])
# 8
image7by7[1, :, 8] = np.array([0, 1, 1, 1, 1, 1, 0])
image7by7[2, :, 8] = np.array([0, 1, 0, 0, 0, 1, 0])
image7by7[3, :, 8] = np.array([0, 0, 0, 1, 1, 1, 0])
image7by7[4, :, 8] = np.array([0, 1, 0, 0, 0, 1, 0])
image7by7[5, :, 8] = np.array([0, 1, 1, 1, 1, 1, 0])
## print('image 1 \n', image7by7)
##
# test data
##
image7by7t = np.zeros([7, 7, 3], float)
image7by7t[1, :, 0] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7t[2, :, 0] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7t[3, :, 0] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7t[4, :, 0] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7t[5, :, 0] = np.array([0, 0, 0, 1, 1, 0, 0])
## print('image 1 \n', image7by7t0)
image7by7t[1, :, 1] = np.array([0, 0, 1, 1, 1, 0, 0])
image7by7t[2, :, 1] = np.array([0, 1, 0, 0, 0, 1, 0])
image7by7t[3, :, 1] = np.array([0, 0, 0, 1, 1, 1, 0])
image7by7t[4, :, 1] = np.array([0, 1, 0, 0, 0, 1, 0])
image7by7t[5, :, 1] = np.array([0, 0, 1, 1, 1, 0, 0])
## print('image 1 \n', image7by7t1)
# test image a 2
image7by7t[1, :, 2] = np.array([0, 0, 1, 1, 1, 1, 0])
image7by7t[2, :, 2] = np.array([0, 1, 0, 0, 1, 0, 0])
image7by7t[3, :, 2] = np.array([0, 0, 0, 1, 0, 0, 0])
image7by7t[4, :, 2] = np.array([0, 0, 1, 0, 0, 0, 0])
image7by7t[5, :, 2] = np.array([0, 1, 1, 1, 1, 1, 0])

# Also one way to 180 degree rotate a matrix
##
# reverse order columns and then rows
##


def M180deg(M):
    return(np.flip(np.flip(M, axis=1), axis=0))


"""also this is a simple, square matrix scipy 2d convolution that you can use to check your algorithm
note: that this algorithm may have some weird scaling (1/sqrt(2pi) would have made sense but it isn’t
quite that), a scalar multiplied by the result, but that shouldn’t matter to the result, the scaling should
be consistent over the elements of the result"""


def conv2Dd_(image, W, stride, Conv):
    if (Conv):
        y = ss.convolve2d(image, W, mode='valid')
    else:
        y = ss.correlate2d(image, W, mode='valid')
    Xdim = len(image[0])//stride
    x = np.zeros([Xdim, Xdim], float)
    for i in range(0, Xdim):
        for j in range(0, Xdim):
            x[i, j] = y[i*stride, j*stride]
            print(x)

W = np.array([[-1.09256581, -0.60177391,  1.24215822],
       [-0.44548082, -1.13145029, -0.54335643],
       [ 0.10972916,  1.96715443,  0.49558543]])

f = np.array([[0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0]])


print(conv2Dd_(f, W, stride=1, Conv=True))

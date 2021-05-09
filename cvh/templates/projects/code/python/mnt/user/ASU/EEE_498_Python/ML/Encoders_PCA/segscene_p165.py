# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:14:50 2020

@author: olhartin@asu.edu
[1]	A. Spizhevoy and A. Rybnikov, OpenCV 3 Computer Vision with Python Cookbook. Packit Birmingham Mumbai, 2018, p. 165.
"""
##
##  import modules
##
import cv2
import numpy as np
import matplotlib.pyplot as plt

##
##  import Caffe model
##
model = cv2.dnn.readNetFromCaffe('../data/fcn8s-heavy-pascal.prototxt','../data/fcn8s-heavy-pascal.caffemodel')

##
##  load the image
##
frame = cv2.imread('../data/scenetext01.jpg')

blob = cv2.dnn.blobFromImage(frame, 1,(frame.shape[1],frame.shape[0]))
model.setInput(blob)
output = model.forward()

##
##  compute the image with per-pixel class labels
##
labels = output[0].argmax(0)
##
##  Visualization
##
plt.figure(figsize=(14,10))
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(frame[:,:,[2,1,0]])
plt.subplot(122)
plt.axis('off')
plt.title('segmentation')
plt.imshow(labels)
plt.tight_layout()
plt.show()

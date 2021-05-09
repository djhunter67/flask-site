# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:03:22 2020

@author: olhartin@asu.edu
[1]	A. Spizhevoy and A. Rybnikov, OpenCV 3 Computer Vision with Python Cookbook. Packit Birmingham Mumbai, 2018, p. 127.
"""
##
##  import modules
##
import cv2
import matplotlib.pyplot as plt
##
##  load test image
##
image = cv2.imread('../data/people.jpg')
##
##  HOG feature detector 
##
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
##
##  detect people in the image
##
locations, weights = hog.detectMultiScale(image)
##
## detect people in bounding boxes
##
dbg_image = image.copy()
for loc in locations:
    cv2.rectangle(dbg_image, (loc[0],loc[1]), (loc[0]+loc[2], loc[1]+loc[3]), (0,255,0), 2)
##
##  visualize results
##
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.title('original')
plt.axis('off')
plt.imshow(image[:,:,[2,1,0]])
plt.subplot(122)
plt.title('detections')
plt.axis('off')
plt.imshow(dbg_image[:,:,[2,1,0]])
plt.tight_layout()
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:17:48 2020

@author: olhartin@asu.edu
[1]	A. Spizhevoy and A. Rybnikov, OpenCV 3 Computer Vision with Python Cookbook. Packit Birmingham Mumbai, 2018, p. 108.
"""
##
##  detecting lines and circles using the Hough transform
##

##
##  import modules
##
import cv2
import numpy as np
import matplotlib.pyplot as plt
##
##  draw a test image
##
print('\n circles and lines drawn at ')
img = np.zeros((500,500),np.uint8)

cv2.line(img, (100,400), (400,350), 255, 3) ## this is the line drawn
print('line points x1,y1 = (100,400)  x2,y2 = (400,350) ')

cv2.circle(img, (200,200), 50, 255, 3)      ## this is the center of circle
print(' circle center at 200,200 radius 50')
##
##  detect lines using the Hough transform
##
print('\n circles and lines detected at ')
lines = cv2.HoughLinesP(img, 1, np.pi/180, 100, 100, 10)[0]
#print('lines ', lines)                      ## line detected
##
##  detect circles using Hough circles
##
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 15, param1=200, param2=30)[0]
#print('circles ', circles)                  ## circle detected
##
##  annotate detecte lines and circles
##
dbg_img = np.zeros( (img.shape[0], img.shape[1], 3), np.uint8)
for x1, y1, x2, y2 in lines:
    print('detected line: ({} {}) ({} {})'.format(x1, y1, x2, y2))
    cv2.line(dbg_img, (x1,y1), (x2,y2), (0,255,0), 2)
    
for c in circles:
    print('Detected circle: center= ({} {}), radius {}'.format(c[0],c[1],c[2]))
    cv2.circle(dbg_img, (c[0], c[1]), c[2], (0,255,0), 2)

##
##  Visualize 
##
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.title('original')
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.title('detected primivitives')
plt.axis('off')
plt.imshow(dbg_img)
plt.show()
"""
"""
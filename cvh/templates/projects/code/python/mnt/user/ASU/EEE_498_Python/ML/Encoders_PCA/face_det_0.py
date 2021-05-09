# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:46:44 2020

@author: olhartin@asu.edu
[1]	A. Spizhevoy and A. Rybnikov, OpenCV 3 Computer Vision with Python Cookbook. Packit Birmingham Mumbai, 2018, p. 168.
"""
##
##  detect faces using a convolution neural network model
##

##
##  import modules
##
import cv2
import numpy as np
##
##  load model and set the confidence threshold
##
model = cv2.dnn.readNetFromCaffe('../data/face_detector/deploy.prototxt','../data/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
CONF_THR = 0.5
##
##  open the video
##
video = cv2.VideoCapture('../data/faces.mp4')
while True:
    ret, frame = video.read()
    if not ret: break
##
##      detect faces in current frame
##
    h, w = frame.shape[0:2]
    blob = cv2.dnn.blobFromImage(frame, 1, (300*w//h, 300), (103,177,123), False)
    model.setInput(blob)
    output = model.forward()
##
##      visualize the results
##
    for i in range(output.shape[2]):
        conf = output[0,0,i,2]
        if conf > CONF_THR:
            label = output[0,0,i,1]
            x0,y0,x1,y1 = (output[0,0,i,3:7] * [w,h,w,h]).astype(int)
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(frame, 'conf: {:,f}'.format(conf),(x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(3)
    if key == 27: break
cv2.destroyAllWindows()
        
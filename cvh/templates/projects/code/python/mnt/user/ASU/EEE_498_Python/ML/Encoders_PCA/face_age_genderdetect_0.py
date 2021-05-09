# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:16:16 2020

@author: olhartin@asu.edu
[1]	A. Spizhevoy and A. Rybnikov, OpenCV 3 Computer Vision with Python Cookbook. Packit Birmingham Mumbai, 2018, p. 169.
"""
##
##  import modules
##
import cv2
import numpy as np
import matplotlib.pyplot as plt
##
##  load modules 
##
age_model = cv2.dnn.readNetFromCaffe('../data/age_gender/age_net_deploy.prototxt', '../data/age_gender/age_net.caffemodel')
gender_model = cv2.dnn.readNetFromCaffe('../data/age_gender/gender_net_deploy.prototxt', '../data/age_gender/gender_net.caffemodel')
##
##  load and crop the source image
##
orig_frame = cv2.imread('../data/face.jpeg')
dx = (orig_frame.shape[1]-orig_frame.shape[0])
orig_frame = orig_frame[:,dx:dx+orig_frame.shape[0]]
##
##  visualize the image
##
plt.figure(figsize=(6,6))
plt.title('original')
plt.axis('off')
plt.imshow(orig_frame[:,:,[2,1,0]])
plt.show()
##
##  load the image with mean pixel values and subtracte them from the source image
##
mean_blob = np.load('../data/age_gender/mean.npy')
frame = cv2.resize(orig_frame, (256,256)).astype(np.float32)
frame -= np.transpose(mean_blob[0], (1,2,0))
##
##  set age and gender lists
##
AGE_LIST = ['(0,2)', '(4,6)','(8,12)','(15,20)','(25,32)','(38,43)','(48,53)','(60,100)']
GENDER_LIST = ['male','female']
##
##  classify gender
##
blob = cv2.dnn.blobFromImage(frame, 1, (256,256))
gender_model.setInput(blob)
gender_prob = gender_model.forward()
gender_id = np.argmax(gender_prob)
print('Gender: {} with prob: {}'.format(GENDER_LIST[gender_id],gender_prob[0, gender_id]))
##
##
##
age_model.setInput(blob)
age_prob = age_model.forward()
age_id = np.argmax(age_prob)
print('Age group: {} with prob: {}'.format(AGE_LIST[age_id],age_prob[0, age_id]))

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:52:16 2020

@author: olhartin@asu.edu
"""
##
##  OpenCV 
##  read an image p.8
import argparse
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--path',default='../data/Lena.png')
params = parser.parse_args()
img = cv2.imread(params.path)
##  did it work
#assert img is not None
print('read()'.format (params.path))
print('shape:', img.shape)
print('dtype:', img.dtype)
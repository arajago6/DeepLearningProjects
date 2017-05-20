#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 00:34:01 2017 using Spyder

@author: arajago6

Build a Convolutional Neural Network to perform binary image classification - The
network would amazingly tell apart pictures of cats and dogs :p

"""

# Install TensorFlow and Keras before proceeding

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

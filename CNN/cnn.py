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

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution with 64 3X3 filters and unit strides on color images resized to 64X64
classifier.add(Convolution2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))

# Step 2 - 2X2 Max Pooling. Strides is same as pool_size. 
classifier.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator



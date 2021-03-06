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

# Adding more convolutional layers to generalize better and get higher test accuracy
classifier.add(Convolution2D(32, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))
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

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        rotation_range = 40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip = True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000,
        nb_epoch = 100,
        validation_data = test_set,
        nb_val_samples=2000)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
print("The image belongs to class " + \
 str(list(training_set.class_indices.keys())[list(training_set.class_indices.values()).index(result[0][0])]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:37:04 2017

@author: rasuishere
"""

# Artificial Neural Network

# Install TensorFlow and Keras before proceeding

# Part 1 - Data preprocessing

# Importing needed libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_ctry = LabelEncoder()
X[:,1] = labelencoder_X_ctry.fit_transform(X[:,1])
labelencoder_X_sex = LabelEncoder()
X[:,2] = labelencoder_X_sex.fit_transform(X[:,2])
# Use one hot encoding to avoid ordinality in country
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Remove first column to avoid dummy variable trap
X = X[:,1:]
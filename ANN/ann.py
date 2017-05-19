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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:41:14 2017 using Spyder

@author: arajago6

Build a Self Organizing Map to detect fraud in acceptance for credit card applications (Statlog UCI dataset)
Uses MiniSom 1.0 package by Giuseppe Vettigli. Liscense: https://creativecommons.org/licenses/by/3.0/

"""

# Importing the needed libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
# We map our 15 feature records into a 10X10 grid
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
# Initializing plot window and plotting inter neuron distances with colors
bone()
pcolor(som.distance_map().T)
# Colorbar gives a legend of color-distance mapping
colorbar()
# Add markers to indicate customers. Red circle signifies rejected applications.
# Green square signifies accepted applications
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    winning_node = som.winner(x)
    # First two parameters help place marker in centre of node 
    plot(winning_node[0] + 0.5, 
         winning_node[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
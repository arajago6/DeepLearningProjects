#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:01:40 2017 using Sypder

@author: arajago6

Build a Recurrent Neural Network to predcit stock prices of Google for a month, 
having trained the RNN using 5 years of historic stock price data

"""

# Part 1 - Data preprocessing

# Importing the needed libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values
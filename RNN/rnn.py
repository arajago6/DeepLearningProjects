#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:01:40 2017 using Sypder

@author: arajago6

Build a Recurrent Neural Network to predict open stock prices of Google for a month, 
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
num_observations = training_set.shape[0]

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the outputs
X_train = training_set[0:num_observations-1]
y_train = training_set[1:num_observations]

# Reshape inputs to match expectation of keras recurrent layer
X_train = np.reshape(X_train,(num_observations-1,1,1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initializing the RNN
regressor = Sequential()

# Adding the input layer and LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compile the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

# Part 3 - Making predictions and visualizing results

# Getting true Google open stock prices for the first month of 2017 
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

# Getting predicted Google open stock prices for the first month of 2017  
inputs = sc.transform(real_stock_price.copy())
inputs = np.reshape(inputs, (inputs.shape[0],1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Open Stock Price')
plt.legend()
plt.show()

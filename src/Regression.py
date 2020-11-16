# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:22:32 2020

@author: Cornellius
"""

#Report 1

import numpy as np
import pandas as pd
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
import sklearn.linear_model as lm
import statsmodels.api as sm 
import pylab as py 
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

#Loading data from .csv
filename = '../data/Real_estate_valuation_data_set.csv'
dataset = pd.read_csv(filename)
raw_data = dataset.to_numpy()
attributeNames = list(dataset)[2:7]

## We skip first two columns (ordinal numbers and transaction date)
cols = range(2, 7) 
X = raw_data[:, cols].astype(np.float)
y = raw_data[:, 7].astype(np.float)

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Standarization
#N=len(y)
#Xc = X - np.ones((N,1))*X.mean(0)
#Xnorm = np.divide(Xc,np.ones((N,1))*np.sqrt(np.square(Xc).mean(0)))

#Regression - Linear Model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(Xnorm,y)
coefficients = model.coef_
y_est = model.predict(Xnorm)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('House price (real)'); ylabel('House price estimated');
subplot(2,1,2)
hist(residual,40)

figure()
plot(X[:, 0], residual, '.r')
xlabel('House age'); ylabel('Residual')

figure()
plot(X[:, 1], residual, '.r')
xlabel('Distance to the nearest MRT station'); ylabel('Residual')

figure()
plot(X[:, 2], residual, '.r')
xlabel('Number of convenience stores in the living circle on foot'); ylabel('Residual')

figure()
plot(X[:, 3], residual, '.r')
xlabel('Latitude'); ylabel('Residual')

figure()
plot(X[:, 4], residual, '.r')
xlabel('Longitude'); ylabel('Residual')


sm.qqplot(residual, line="s")
py.show()

show()

# Regression - cross-validation

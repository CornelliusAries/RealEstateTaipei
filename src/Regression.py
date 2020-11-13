# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:22:32 2020

@author: Cornellius
"""

#Report 1

import numpy as np
import pandas as pd
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm
import statsmodels.api as sm 
import pylab as py 
import scipy.stats as stats


filename = '../data/Real_estate_valuation_data_set.csv'
dataset = pd.read_csv(filename)
raw_data = dataset.to_numpy()

## We skip first two columns (ordinal numbers and transaction date)
cols = range(2, 7) 
X = raw_data[:, cols].astype(np.float)
Y = raw_data[:, 7].astype(np.float)

#Standarization
N=len(Y)
Xc = X - np.ones((N,1))*X.mean(0)
Xnorm = np.divide(Xc,np.ones((N,1))*np.sqrt(np.square(Xc).mean(0)))

#Regression_1
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(Xnorm,Y)
coefficients = model.coef_
y_est = model.predict(Xnorm)
residual = y_est-Y


# Display scatter plot
figure()
subplot(2,1,1)
plot(Y, y_est, '.')
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
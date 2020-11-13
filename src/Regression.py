# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:22:32 2020

@author: Cornellius
"""

#Report 1

import numpy as np
import pandas as pd
from matplotlib.pyplot import boxplot, xticks, ylabel, title, show
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)

filename = 'Real_estate_valuation_data_set.csv'
dataset = pd.read_csv(filename)
raw_data = dataset.to_numpy()
## We skip first two columns (ordinal numbers and transaction date)
cols = range(2, 8) 
X = raw_data[:, cols].astype(np.float)

statValues = np.empty([6,4])
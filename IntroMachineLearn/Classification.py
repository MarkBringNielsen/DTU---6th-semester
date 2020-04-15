# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:17:05 2020

@author: Mark
"""

# exercise Load data
import numpy as np
import pandas as pd

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show, hist, ylim)
from scipy.stats import zscore
from scipy.linalg import svd
import matplotlib.pyplot as plt
import torch

from matplotlib.pylab import (semilogx, loglog, 
                           title, grid)


import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from toolbox_02450 import train_neural_net, draw_neural_net

import scipy.stats as st

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier

#######################################################
### Load the Wine csv data using the Pandas library ###
#######################################################
filename = 'wine.csv'
df = pd.read_csv(filename, header=None)

X = df.drop(0, axis=1).values
y = df[0]


N, M = X.shape

## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = 10
K2 = 10
L = 40
CV = model_selection.KFold(n_splits=K1)
errors = np.zeros((N,L))


k=0
iii = 0
for train_index, test_index in CV.split(X,y):
    print('iii is:')
    print(iii) 
    

        
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    i=0
    for train_index, test_index in CV.split(X_train, y_train):
        print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
    
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l)
            knclassifier.fit(X_train, y_train)
            y_est = knclassifier.predict(X_test)
            e = np.sum(y_est[0]!=y_test.values[0])
            errors[i,l-1] = e
    
        i+=1
        
        #Train the best model and test it
    print(errors)    
    # Plot the classification error rate
    figure()
    plot(100*sum(errors,0)/N)
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    show()
    
    
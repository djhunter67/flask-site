# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 07:33:13 2018

@author: olhartin@asu.edu
Homework 2 companion
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
## observations for 3 variables x0, x1, x2
np.random.seed(1)
##X = np.array([
## fill in data here for hard coding
##])
##
##  this is a single classificaiton
##
## target
##y = np.array([ fill in target for hard coding])
##
##      Read from CSV
##
Nfeatures = 3
ald = pd.read_csv('Dataset_1.csv')
cols = ald.columns
X = ald.iloc[:,0:Nfeatures].values
y = ald.iloc[:,Nfeatures].values
##
##      Scale input
##
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X[:,0:Nfeatures])
##
##      convert to compact form, now in compact form
##      the first weight is the intercept
##
Fones = np.ones(len(X_std[:,0]),dtype=float)
X_std_ = np.column_stack((Fones,X_std))
##
##      training and test datasets
##
## split the problem into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std_,y,test_size=0.3,random_state=0)
##
##      Models, cost functions, derivatives, activation functions
##      Fill in
##
def model(X,w):
    ...

##
##      Activation
##
def activation(Z):
    if (Z >= 0.0): return(1)
    else: return(-1)
##
##  Aldine
##
def aldine_sgd(X, w, Y, eta):
    epochs = 1500
    errors = []
    total_error = 1.0
    tol = 0
    t = 1
    while (t<epochs and total_error>tol):   # stop when the error is 0, or limit in epochs
        total_error = 0.0
        dw = 0.0
##
##      fill in your code here 
##
        t += 1
    return w,errors,t       ## return weight, total error for each epoch, and last epoch
##
##      apply model
##
def apply_model(X,w,y):
    ypreds = []
    error_total = 0.0
    for i, x in enumerate(X):
        ypred = activation(model(x,w))
##      we want 1 and -1 not 1 and 0 so adjust limits
        error = y[i]-ypred
##        print(' predicted ', ypred, ' actual ', y_train[i], ' error ', error)
        error_total += error*error
        ypreds.append(ypred)
    print(' Total Error ', error_total)
##
##      determine accuracy of prediction of the training set
##
    print('Misclassified samples: %d' % (y != ypreds).sum())
    print('Accuracy: %.2f' % accuracy_score(y, ypreds))
    return ypreds,error_total
##
##      run Aldine
##
w = np.ones(len(X_train[0]))*0.1          ## Initial Weight
##  very dependent on the learning rate and initial weight
w,errors,t = aldine_sgd(X_train,w,y_train,0.001) ## standarized input, initial weight, target, learning rate
print(' Weights ', w)
print(' Epochs ', t)
##
##      plot errors per iteration 
##
plt.plot(errors)
plt.title('Error vs Iteration')
plt.xlabel('iterations')
plt.ylabel('error')
plt.grid()
plt.show()
##
##      look at results from model
##
ypreds_train,error_total = apply_model(X_train,w,y_train)
##
##      apply model seights to the test set
##
ypreds_test,error_total = apply_model(X_test,w,y_test)
##
##      Test with Perceptron
##
##      we don't need the compact model since perceptron
##      handles the intercept
##
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=0)
## perceptron linear
from sklearn.linear_model import Perceptron
## forward pass calculation of decision function
## backward update weights
## epochs are number of times the alg sees entire dataset
## iteration 
## an epoch is one forward and a backward pass of all training samples (also an iteration)
## no of iterations or epochs, and rate of convergence eta0
## ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
## max_iter, tol, if it is too low it is never achieved
## and continues to iterate to max_iter when above tol
## fit_intercept, fit the intercept or assume it is 0
## slowing it down is very effective, eta is the learning rate
ppn = Perceptron(max_iter=1500, tol=1e-6, eta0=0.001, random_state=0)
ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_train)
print('Misclassified samples: %d' % (y_train != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_train, y_pred))

y_pred = ppn.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
##
##  comparison of results between methods
##
print(' Weights ', w)
print('perceptron weights ', ppn.intercept_, ppn.coef_ )
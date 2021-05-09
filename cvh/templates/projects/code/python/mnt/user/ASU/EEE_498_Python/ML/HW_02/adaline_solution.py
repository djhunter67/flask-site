from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('agg')


##
# ALDINE, Quantization, LS Costfunction, SGD, read from file
# standard normalization, compact notation
# Dataset 1
# observations for 3 variables x0, x1, x2
np.random.seed(1)

Nfeatures = 3
ald = pd.read_csv('/home/djhunter67/winhome/Documents/ASU/EEE_498_Python/ML/HW_02/Dataset_1.csv')
cols = ald.columns
X = ald.iloc[:, 0:Nfeatures].values
y = ald.iloc[:, Nfeatures].values
Nobs = len(X[:, 0])
##
# Scale input
##
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X[:, 0:Nfeatures])  # normalization
# X_std = X.copy()                                    ## no normalization
##
# convert to compact form, now in compact form
# the first weight is the intercept
##
Fones = np.ones(Nobs, dtype=float)
X_std_ = np.column_stack((Fones, X_std))
##
# training and test datasets
##
# split the problem into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_std_, y, test_size=0.3, random_state=0)
##
# Model y = X dot w
##


def model(X, w):
    return(np.dot(X, w))
##
# Adaline SSE cost function
##


def CostF(X, w, Y):
    return((model(X, w)-Y)**2)  # positive if model and Y not same
##
# Derivative of SSE Cost Function
##


def dCostF(X, w, Y):
    return((2.*model(X, w)-Y)*X)
##
##
##


def aCostF(X, w, Y):
    error = activation(model(X, w))-Y
    return(error*error)
##
# Activation
##


def activation(Z):
    if (Z >= 0.0):
        return(1)
    else:
        return(-1)
##
# Aldine
##


def aldine_sgd(X, w, Y, eta):
    Nobs = len(X[:, 0])
    epochs = 1500
    errors = []
    total_error = 1.0
    tol = 0
    t = 1
    while (t < epochs and total_error > tol):   # stop when the error is 0, or limit in epochs
        total_error = 0.0
        dw = 0.0
        for i in range(int(len(X))):
            # Stochastic Gradient Descents (SGD), choose method
            # create a random number pointing to data
            j = np.random.randint(Nobs)
# j = i                         # all the data in the dataset
            error = aCostF(X[j], w, Y[j])      # error from Cost function
# error = aCostF(X[j],w,Y[j])   # error from Cost function, using activation
            total_error += error            # total error from all cases
            if (error != 0.0):  # this activation step causes it to stop at the solution
                # X[i] is the weight update Y[i] is the direction
                dw += dCostF(X[j], w, Y[j])

        # it only compares to 0 so a constant scale doesn't matter
        w -= eta*dw/(len(X))
        if (t % 100 == 0):
            print(' epoch ', t, 'total_error ', total_error)
        errors.append(total_error/len(X))  # error from each epoch
        t += 1
##    print(' final weights ', w)
    return w, errors, t  # return weight, total error for each epoch, and last epoch
##
# apply model
##


def apply_model(X, w, y):
    ypreds = []
    error_total = 0.0
    for i, x in enumerate(X):
        ypred = activation(model(x, w))
# we want 1 and -1 not 1 and 0 so adjust limits
        error = y[i]-ypred
##        print(' predicted ', ypred, ' actual ', y_train[i], ' error ', error)
        error_total += error*error
        ypreds.append(ypred)
    print(' Total Error ', error_total)
    ##
    # determine accuracy of prediction of the training set
    ##
    print('Misclassified samples: %d' % (y != ypreds).sum())
    print('Accuracy: %.2f' % accuracy_score(y, ypreds))
    return ypreds, error_total


##
# run Adaline
##
w = np.ones(len(X_train[0]))*0.1  # Initial Weight
# very dependent on the learning rate and initial weight
# without normalization learning rate 0.00001
# with normalization learning rate 0.001 or even 0.01
# 0.1 or greater converges quickly but to increased error
# standarized input, initial weight, target, learning rate
w, errors, t = aldine_sgd(X_train, w, y_train, 0.001)
print(' Weights ', w)
print(' Epochs ', t)
##
# plot errors per iteration
##
plt.plot(errors)
plt.title('Error vs Iteration')
plt.xlabel('iterations')
plt.ylabel('error')
plt.grid()
plt.savefig('adaline_solution')
##
# look at results from model
##
ypreds_train, error_total = apply_model(X_train, w, y_train)
##
# apply model seights to the test set
##
ypreds_test, error_total = apply_model(X_test, w, y_test)

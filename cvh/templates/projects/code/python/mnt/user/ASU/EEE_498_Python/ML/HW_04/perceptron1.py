# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 07:33:13 2018

@author: olhartin@asu.edu
"""
## https://maviccprp.github.io/a-perceptron-in-just-a-few-lines-of-python-code/
## implement fusion p99 100 MLR Watt
import numpy as np
import matplotlib.pyplot as plt
##
##  multi classification linear
##
## observations for 3 variables x0, x1, x2
## 3 features
X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
    [6, 1, -1],
])
## target 3 possible classifications
Y = np.array([
    [1, -1, -1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, 1],
])
## argmax
def argmax(V,m):
    maxV = np.max(V)
    VV = np.copy(V)
    j=0
    for v in V:
        if (v==maxV and v!=0.0):
            VV[j] = 1.0
            j+=1
        else:
            VV[j] = -1.0
            j+=1
    return(VV)
##      rescale from 0 or 1 to -1 or 1
def rescale(V):
##    V = V*2-1
    return(V)
##  our perceptron - stochastic gradient descent
def perceptron_sgd(X, Y, eta):
    samples = len(Y[:,0])
    classifications = len(Y[0])
    features = len(X[0])
    print(' Samples ', samples, ' Classifictions ', classifications, ' Features ', features)
##      weight for each feature
##      and each classificaition type
    w = np.zeros((features,classifications),float)
    epochs = 20
    errors = []
    error = np.zeros(classifications,float)
    for t in range(epochs):
        total_error = 0.0
##      each sample or observation
        for i, x in enumerate(X):
##              will be neg if there is a missclassification
##              x dot w is the pred of y, times y give pos or neg sign 
##              to drive the weight higher or lower
##              multi classification through fusion
            for j in range(classifications):
                error[j] = np.dot(X[i],w[j])
##              error vector
            error = argmax(error,1)
##              error will be 1 for the classification
##                -1 for the missed classification
            for j in range(classifications):
                error[j] *= rescale(Y[i,j])
                if (error[j] <= 0):
                    dw = X[i] * Y[i,j]
                    w[j] += eta*dw
                    total_error += error[j]
##                  print(' Epoch ', t, ' dw ', dw,' w ', w, ' error ', error)
##        print(' epoch ', t,' w ', w,'\n')
##        for i, x in enumberate(X)
        errors.append(total_error*-1)
    print(' Epochs ', t) 
    print(' eta ', eta)
    plt.plot(errors)
    plt.title('Error vs Iteration')
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.grid()
    plt.show()
    print(' final weights \n', w)

    return w
##
##
classifications = len(Y[0])
samples = len(Y[:,0])
## run perceptron
W = perceptron_sgd(X,Y,1)
## Binarizer creates a transform that binary at threshold, in this case 0.0
Ypred = np.zeros((samples,classifications),float)
error_total = 0.0
for i, x in enumerate(X):
    for j in range(classifications):
##        print(i,X[i],j,W[j])
        Ypred[i,j] = np.dot(X[i],W[j])
##        print(i,j,Ypred[i,j],Y[i])
    Ypred[i] = argmax(Ypred[i],1)
    error = Ypred[i] - Y[i]
    error_total += error
    print(' Sample ', i, ' Error ', error)
print(' Total Error ', error_total)
print(W)
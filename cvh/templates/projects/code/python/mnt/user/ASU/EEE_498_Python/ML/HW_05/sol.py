import math
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 0:4]  # from this only take features 0,1,2,3
y = iris.target


def PGauss(mu, sig, x):
    var=float(sig)**2
    denom=(2*math.pi*var)**0.5
    num=math.exp(-(float(x)-float(mu))**2/(2*var))
    return(num/denom)


##
# pull data from iris dataset
##
iris=datasets.load_iris()
# X_ = iris.data[:,[2,3]]   ## from this only take features 2 and 3
X=iris.data[:, 0:4]  # from this only take features 0,1,2,3
y=iris.target
##
# Extract the mean and standard deviation for each feature
# for each classification
##
Nobs=len(X[:, 0])
Nfeatures=len(X[0, :])
Cmax=max(y)  # assumes classes are numbered 0,1,2 ...
u=np.zeros((Nfeatures, Cmax+1), float)
s=np.zeros((Nfeatures, Cmax+1), float)
for c in range(Cmax+1):
    for f in range(Nfeatures):
        # means f where classificaiton is c
        u[f, c]=X[np.where(y == c), f].mean()
        # means f where classificaiton is c
        s[f, c]=X[np.where(y == c), f].std()
        print(' feature ', f, ' classification ', c,
              ' mean ', u[f, c], ' stdev ', s[f, c])
##
# function which determines prob of c given X  P(c|X)
# using Naive Bayes, and return the most likely c for X
##


def probatx(X):
    P = np.ones((Cmax+1), float)
    Pcmax = 0.0
    Clmax = 0
    for c in range(Cmax+1):
        Pc = y.tolist().count(c)/Nobs  # P(c)
        for f in range(Nfeatures):
            P[c] *= PGauss(u[f, c], s[f, c], X[f])
# print('c ', c,'n ', n, 'x ', X[n,f], 'prob ',P[c,n])
        P[c] *= Pc
        if (P[c] > Pcmax):
            Pcmax = P[c]
            Clmax = c
# print('c ', c,'n ', n, 'prob ',P[c,n])
# print('Pcmax ', Pcmax, ' C ', Cmax)
# print(P)
    return(Pcmax, Clmax, P)


##
# make predictions,
##
ypred = np.zeros(Nobs, int)
for n in range(Nobs):
    prob, C, PP = probatx(X[n, :])
    ypred[n] = C
# if (C!=y[n]):
# print('c ', C, 'Prob ', PP[C], 'y[n] ', y[n], 'Prob y ', PP[y[n]])
# print('n ', n, 'prob ', prob, 'C ', C, 'y ', y[n])
##
# Results, number of missclassifications, and accuracy
##
print('Train Accuracy: %.4f' % accuracy_score(y, ypred))
print('Number in trained ', len(y))
print('Number of features ', len(X[0]))
print('Misclassified samples: %d' % (y != ypred).sum())
# where were there errors
err = np.where(y != ypred)
print('errors at indices ', err, 'actual classificiton ',
      y[err], ' pred myNB ', ypred[err])

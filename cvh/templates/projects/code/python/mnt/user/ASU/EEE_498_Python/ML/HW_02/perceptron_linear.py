

from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from numpy import random, ones, column_stack, dot
from pandas import read_csv
from sklearn.preprocessing import StandardScaler


random.seed(1)

Nfeatures = 3
data_ = read_csv(
    '/home/djhunter67/winhome/Documents/ASU/EEE_498_Python/ML/HW_02/Dataset_2.csv')
cols = data_.columns

X = data_.iloc[:, 0:Nfeatures].values
y = data_.iloc[:, Nfeatures].values
lobes = len(X[:, 0])
standard_scaler = StandardScaler()
# Normalization Step
#
X_std = standard_scaler.fit_transform(X[:, 0:Nfeatures])

first_ones = ones(lobes, dtype=float)
X_std_ = column_stack((first_ones, X_std))

X_train, X_test, y_train, y_test = train_test_split(
    X_std_, y, test_size=0.3, random_state=0)

# perceptron linear
print("\nLinear Perceptron SKlearn")

ppn = Perceptron(max_iter=1500, tol=1e-6, eta0=0.001, random_state=0)
ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_train)
print('Misclassified samples: %d' % (y_train != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_train, y_pred))

y_pred = ppn.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

##
# comparison of results
##
# print(' Weights ', w)
print('perceptron weights ', ppn.intercept_, ppn.coef_)

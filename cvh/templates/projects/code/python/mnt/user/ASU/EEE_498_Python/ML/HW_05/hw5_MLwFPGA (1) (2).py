import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:,0:4]
y = iris.target
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

gauss_nb= GaussianNB()
gauss_nb.fit(X_train_std, y_train)

y_predict = gauss_nb.predict(X_test_std)
miss = 0
for i in range (len(y_test)):
    if (y_predict[i] != y_test[i]):
        print("predicted:", y_predict[i]," actual: ", y_test[i])
        miss += 1
    else:
        miss = miss
"""
"""       
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)
"""
gauss_nb= GaussianNB()
gauss_nb.fit(X, y)

y_predict = gauss_nb.predict(X)
miss=0
for i in range (len(y)):
    if (y_predict[i] != y[i]):
        print("predicted:", y_predict[i]," actual: ", y[i])
        miss += 1
    else:
        miss = miss 
print("Number of Missclassified: ", miss)

accuracy = 1- (miss/len(y))
print('Accuracy: ', accuracy)
print('Accuracy: ' , accuracy_score(y, y_predict))
 


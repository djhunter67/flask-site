# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:20:12 2020

@author: Samc
"""
##
##  convert to multiclass one hot coding
##
def onehoty(y):
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = y.reshape(len(y), 1)
    print('2',integer_encoded)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    return(onehot_encoded)
##
##  Most Highly Correlated
##
def mosthighlycorrelated(mydataframe, numtoreport): 
# find the correlations 
    cormatrix = mydataframe.corr() 
# set the correlations on the diagonal or lower triangle to zero, 
# so they will not be reported as the highest ones: 
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T 
# find the top n correlations 
    cormatrix = cormatrix.stack() 
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index() 
# assign human-friendly names 
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"] 
    return cormatrix.head(numtoreport)


##
##      convert to pandas dataframe
##
import pandas as pd
irisp = pd.DataFrame(iris.data,columns=iris.feature_names)

print("Most Highly Correlated")
print(mosthighlycorrelated(irisp,Nfeatures))

print('\n',irisp.head())


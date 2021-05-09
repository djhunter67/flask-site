# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 12:41:01 2021

@author: creyes
"""

import os
import cv2
import numpy as np 
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
#from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#load the dataset
data = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')
#show its head
data.head()

#rename the class column
data.rename(columns={'0':'label'}, inplace=True)
#show the top 5 rows 
data.head()
# how many labels do we have 
data.label.nunique()

# Split data to Features X and labels y
X = data.drop('label',axis = 1)
y = data.label
#get the shape of labels and features 
print(f'Features SHAPE :{X.shape}')
print(f'Class Column SHAPE :{y.shape}')
#split into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# scale data
scaler = MinMaxScaler()
scaler.fit(X_train)
#scaling data 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train[1:10]

X_train = np.reshape(X_train, (X_train.shape[0], 28,28,1)).astype('float32')
X_test = np.reshape(X_test, (X_test.shape[0], 28,28,1)).astype('float32')
print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)

y_train = np_utils.to_categorical(y_train,num_classes=26,dtype=int)
y_test = np_utils.to_categorical(y_test,num_classes=26,dtype=int)
y_train.shape,y_test.shape

#define a mapping dict 
letters_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',
             7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
             14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
             21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
#show 
fig, axis = plt.subplots(3, 3, figsize=(20, 20))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_train[i].reshape(28,28))
    ax.axis('off')
    ax.set(title = f"Alphabet : {letters_dict[y_train[i].argmax()]}")
    
# count by label
sb.set_style('whitegrid')
df=data.copy()
df['label'] = df['label'].map(letters_dict)

labels_count = df.groupby('label').size()
labels_count.plot.bar(figsize=(15,10))
plt.ylabel("Count")
plt.xlabel("Alphabets")
plt.show()


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
#output layer 
model.add(Dense(26,activation ="softmax"))
#compile 
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#model summary
model.summary()    

import time 
start=time.time()
history = model.fit(X_train, y_train, epochs=10,batch_size=128,verbose=2,validation_data = (X_test,y_test))
end=time.time()
print('\n')
print(f'Execution Time :{round((end-start)/60,3)} minutes')


## show loss and accuracy scores 
scores =model.evaluate(X_test,y_test,verbose=0)
print('Validation Loss : {:.2f}'.format(scores[0]))
print('Validation Accuracy: {:.2f}'.format(scores[1]))


# Plot training loss vs validation loss 
plt.figure()
fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,7))
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.legend(['training','validation'])
ax1.set_title('Loss')
ax1.set_xlabel('epochs')
## plot training accuracy vs validation accuracy 
ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.legend(['training','validation'])
ax2.set_title('Acurracy')
ax2.set_xlabel('epochs')


# Plot the predictions 
preds = model.predict(X_test)
X_test_ = X_test.reshape(X_test.shape[0], 28, 28)
fig, axis = plt.subplots(3, 3, figsize=(20, 20))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test_[i])
    ax.axis('off')
    ax.set(title = f"Real Alphabet : {letters_dict[y_test[i].argmax()]}\nPredicted Alphabet : {letters_dict[preds[i].argmax()]}");


#get the predicted alphabets 
predicted_values = [np.argmax(y, axis=None, out=None) for y in preds]
#get the alphabets using letters dictionnary 
predicted_alphabets =[letters_dict[i] for i in predicted_values]
#Reverse y_test from one hot encoder to an array 
test_labels = [np.argmax(y, axis=None, out=None) for y in y_test]
#same for real alphabets 
test_alphabets = [letters_dict[i] for i in test_labels]
# create a submission dataframe 
submission = pd.DataFrame({'Real Alphabet':test_alphabets,'Predicted Aplhabet':predicted_alphabets })
# save to a csv file 
submission.to_csv('submission.csv', index=False)
print(" Submission  successfully saved!")




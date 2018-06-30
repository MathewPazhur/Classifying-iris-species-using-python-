# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 15:33:05 2018

@author: Mathew
"""

#importing libraries

import numpy as np
from scipy import sparse
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

## Preparing data

#importing iris dataset from skikit learn

from sklearn.datasets import load_iris
iris_dataset=load_iris()

#displaying list of keys 

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

#displaying value of DESCR key

print(iris_dataset['DESCR'][:1000]+"\n...")

#displaying value of target_names key

print("Target names : {}".format(iris_dataset['target_names']))

#displaying values of features key

print("Feature names: {}".format(iris_dataset['feature_names']))

#displaying type of data

print("Type of data: {}".format(type(iris_dataset['data'])))

#displaying shape of data 

print("Shape of data : {}".format(iris_dataset['data'].shape))

#display first 5 columns of data

print("First five columns of data : \n{}".format(iris_dataset['data'][:5]))

#displaying type of target 

print("Type of target : {}".format(type(iris_dataset['target'])))

#displaying shape of target

print("Shape of target : {}".format(iris_dataset['target'].shape))

#displaying the target data

print("Target : {}".format(iris_dataset['target']))



#----------------------------------------------------------------------------------------------------------------------



## Splitting the data into test and train data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

#Here random state is the seed used to shuffle the dataset
#X_train and X_test are for data and y_train, y_test are for labels

#Displaying shape of above mentioned arrays

print("X_train shape : {}".format(X_train.shape))
print("X_test shape : {}".format(X_test.shape))
print("y_train shape : {}".format(y_train.shape))
print("y_test shape : {}".format(y_test.shape))


#----------------------------------------------------------------------------------------------------------------------


## Visualizing Data

#Create dataframe from data in X_train
#Label the columns using the strings in iris_dataset.feature_names

iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)

#create a scatter matric from the dataframe, color by y_train

grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)


#----------------------------------------------------------------------------------------------------------------------------------------

## Training

# k - nearest neighbours

#instantiating KNeighborsClassifier class into an object

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

#training data

knn.fit(X_train, y_train)

#------------------------------------------------------------------------------------------------------------------------------------

## Making Predictions

#Shape of new flower

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

#Making the prediction

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predictied target name : {}".format(iris_dataset['target_names'][prediction]))

#----------------------------------------------------------------------------------------------------------------------

## Evaluating the model

y_pred = knn.predict(X_test)
print("Test set prediction: \n {}".format(y_pred))

#Measuring accuracy

print("Test set score : {:.2f}".format(np.mean(y_pred==y_test)))

#Using score method to measure accuracy

print("Test set score : {:.2f}".format(knn.score(X_test, y_test)))

#----------------------------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:40:21 2019

@author: Ben
"""

# knn model

# Importation des packages

import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from sklearn.model_selection import GridSearchCV
import pickle


os.chdir("C:/Users/Ben/Supercase/My branch/")


#%%
# Lecture des données
ids = []
labels = []
for i in range(11):
    directory = 'data/%i/' %i
    for j in os.listdir(directory):
        ids.append(directory+j)
        labels.append(i)
        
digits_data = pd.DataFrame(list(zip(ids,labels)))
X =  digits_data.iloc[:,0]
y =  digits_data.iloc[:,1]
               


#%%
# Cross validation
temp = []
for sample in X:
    img = Image.open(sample)    
    img = np.array(img)
    img.resize((100,256))
#    img = img.reshape((img.shape[0],img.shape[1],1))
    img = img.reshape((img.shape[0] * img.shape[1]))
    temp.append(img)
    
X = np.asarray(temp)



#%%
# Grid search for hyperparameter otpimization n_neighbors with 5-CV validation

#create new a knn model
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {"n_neighbors": np.arange(1,10)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)

#%%

#check top performing n_neighbors value
n_neighbors = knn_gscv.best_params_

#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_

#%%
#Final model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {"n_neighbors": np.arange(n_neighbors,n_neighbors + 1)}
#use gridsearch to test all values for n_neighbors
knn_gscv2 = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv2.fit(X, y)

#%%
# save the model to disk
filename = 'knn_model.sav'
pickle.dump(knn_gscv2, open(filename, 'wb'))
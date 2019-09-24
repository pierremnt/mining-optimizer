# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:35:11 2019

@author: Ben
"""

import numpy as np
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dropout, Activation
from keras.layers import BatchNormalization
import tensorflow as tf
from keras import regularizers
import keras.backend
from keras.optimizers import Adam
#from keras.utils import plot_model
#from keras.callbacks import TensorBoard,EarlyStopping

import matplotlib.pyplot as plt
from PIL import Image

from keras.models import load_model

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
# Séparation training test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=2)

# Flattening des images
temp = []
for sample in X_train:
    img = Image.open(sample)
    img = np.array(img)
    img.resize((100,256))
    img = img.reshape((img.shape[0],img.shape[1],1))
    temp.append(img)
    
X_train = np.asarray(temp)

temp = []
for sample in X_test:
    img = Image.open(sample)
    img = np.array(img)
    img.resize((100,256))
    img = img.reshape((img.shape[0],img.shape[1],1))
    temp.append(img)
    
X_test = np.asarray(temp)

#%%
# Model definition

model_input = Input((100, 256, 1))

x = Conv2D(32, (3, 3), padding='same', name='conv2d_hidden_1', kernel_regularizer=regularizers.l2(0.01))(model_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_1')(x)
x = Dropout(0.30)(x)

x = Conv2D(63, (3, 3), padding='same', name='conv2d_hidden_2', kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_2')(x)
x = Dropout(0.30)(x)

x = Conv2D(128, (3, 3), padding='same', name='conv2d_hidden_3', kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_3')(x)
x = Dropout(0.30)(x)

x = Flatten()(x)

x = Dense(1024, activation ='relu', kernel_regularizer=regularizers.l2(0.01))(x)

output = Dense(output_dim = 11, activation = 'softmax', name='output')(x)

model = keras.models.Model(input = model_input , output = output)
model._make_predict_function()

#%%
# train
lr = 1e-3
epochs = 10

optimizer = Adam(lr=lr, decay=lr/200)
model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics = ['accuracy'])
keras.backend.get_session().run(tf.initialize_all_variables())
history = model.fit(X_train, y_train, batch_size= 30, nb_epoch=epochs, verbose=1, validation_data=(X_test, y_test))

#%%
# Plot accuracy
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=0.5)
plt.plot(history.history['val_accuracy'],'b',linewidth=0.5)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves Digit',fontsize=16)
plt.show()


#%%
# Plot loss
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=0.5)
plt.plot(history.history['val_loss'],'b',linewidth=0.5)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves Digit',fontsize=16)
plt.show()


#%%
# save model and architecture to single file
model.save("model.single_digit_fulltrain")
print("Saved model to disk")

#%%
# Predict
y_pred = model.predict(X_test)
pred_list = [np.argmax(y_pred[i]) for i in range(len(y_pred))]

correct_preds = sum( pred_list == y_test)
print('exact accuracy', correct_preds / len(y_pred))

#%%
def model_prediction(image_list,model = "model.single_digit_fulltrain"):
    model = model.load(model)
    temp = []
    for img in image_list:
        img.resize((100,256))
        img = img.reshape((img.shape[0],img.shape[1],1))
        temp.append(img)
    X_test = np.asarray(temp)
    y_pred = model.predict(X_test)
    return [np.argmax(y_pred[i]) for i in range(len(y_pred))]

def calc_accuracy(dic, model = "model.single_digit_fulltrain"):
    correct_preds = 0
    n = len(dic)
    model = model.load(model)
    for img,label in dic.items():
        img.resize((100,256))
        img = img.reshape((img.shape[0],img.shape[1],1))
        test = np.asarray(img)
        pred = np.argmax(model.predict(test))
        correct_preds += (label == pred)
    print('exact accuracy', correct_preds /n)
    

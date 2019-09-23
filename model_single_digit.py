# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:01:44 2019

@author: Ben
"""
# Single digits model

# Importation des packages

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from keras.models import load_model

os.chdir("C:/Users/Ben/Supercase/GasPumpOCR-master/training")
#%%
# Lecture des données

# The full `train` split.
#load mnist data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

def create_mnist_dataset(data, labels, batch_size):
  def gen():
    for image, label in zip(data, labels):
        yield image, label
  ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28,28 ), ()))

  return ds.repeat().batch(batch_size)

#train and validation dataset with different batch size
train_dataset = create_mnist_dataset(X_train, y_train, 10)
valid_dataset = create_mnist_dataset(X_test, y_test, 20)

#%% 
# Séparation training test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

temp = []
#for sample in X_train:
for img in X_train:
#    img = Image.open("C:/Users/Ben/Supercase/GasPumpOCR-master/training/" + sample)    
    img = np.array(img)
#    img.resize((28,28))
    img = img.reshape((img.shape[0],img.shape[1],1))
    temp.append(img)
    
X_train = np.asarray(temp)

temp = []
#for sample in X_test:
for img in X_test:
#    img = Image.open("C:/Users/Ben/Supercase/GasPumpOCR-master/training/" + sample)
    img = np.array(img)
#    img.resize((28,28))
    img = img.reshape((img.shape[0],img.shape[1],1))
    temp.append(img)
    
X_test = np.asarray(temp)


#%%
# Model definition

model_input = Input((28, 28, 1))
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

output = Dense(output_dim = 10,activation = 'softmax', name='output')(x)

model = keras.models.Model(input = model_input , output = output)
model._make_predict_function()

#%%
# train
lr = 1e-3
epochs = 10

optimizer = Adam(lr=lr, decay=lr/10)
model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics = ['accuracy'])
keras.backend.get_session().run(tf.initialize_all_variables())
history = model.fit(X_train, y_train, batch_size= 30, nb_epoch=epochs, verbose=1, validation_data=(X_test, y_test))

#%%
# Plot acc
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
model.save("model.mnist")
print("Saved model to disk")

#%%
# Predict
            
y_pred = model.predict(X_test)
pred_list = [np.argmax(y_pred[i]) for i in range(len(y_pred))]

correct_preds = sum( pred_list == y_test)
print('exact accuracy', correct_preds / len(y_pred))

#%%
# load model
model = load_model('model.mnist')

for layer in model.layers[:-4]:
    layer.trainable = False

#%%
ids = []
labels = []
for i in range(10):
    directory = '%i/' %i
    for j in os.listdir(directory):
        ids.append(directory+j)
        labels.append(i)
        
digits_data = pd.DataFrame(list(zip(ids,labels)))
X =  digits_data.iloc[:,0]
y =  digits_data.iloc[:,1]

#%%
ids = []
labels = []
for i in range(10):
    directory = '%i/' %i
    for j in os.listdir(directory):
        ids.append(directory+j)
        labels.append(i)
digits_data = pd.DataFrame(list(zip(ids,labels)))

X =  digits_data.iloc[:,0]
y =  digits_data.iloc[:,1]

#%% 
# Séparation training test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

temp = []
for sample in X_train:
    img = Image.open("C:/Users/Ben/Supercase/GasPumpOCR-master/training/" + sample)
    img = np.array(img)
    img.resize((28,28))
    img = img.reshape((img.shape[0],img.shape[1],1))
    temp.append(img)
    
X_train = np.asarray(temp)

temp = []
for sample in X_test:
    img = Image.open("C:/Users/Ben/Supercase/GasPumpOCR-master/training/" + sample)
    img = np.array(img)
    img.resize((28,28))
    img = img.reshape((img.shape[0],img.shape[1],1))
    temp.append(img)
    
X_test = np.asarray(temp)

#%%
# train
lr = 1e-3
epochs = 10

optimizer = Adam(lr=lr, decay=lr/10)
model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics = ['accuracy'])
keras.backend.get_session().run(tf.initialize_all_variables())
history = model.fit(X_train, y_train, batch_size= 30, nb_epoch=epochs, verbose=1, validation_data=(X_test, y_test))

#%%
# Plot acc
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
model.save("model.final")
print("Saved model to disk")

#%%
# Predict
            
y_pred = model.predict(X_test)
pred_list = [np.argmax(y_pred[i]) for i in range(len(y_pred))]

correct_preds = sum( pred_list == y_test)
print('exact accuracy', correct_preds / len(y_pred))




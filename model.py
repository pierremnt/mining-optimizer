# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:01:44 2019

@author: Ben
"""
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
from keras.utils import plot_model
from keras.callbacks import TensorBoard,EarlyStopping
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

os.chdir("C:/Users/Ben/Supercase/to_commit/mining-optimizer")
#%%
# Lecture des données digitales

suffix = "HQ_digital.csv"
csv_directory = 'data/'
csv_files = [i for i in os.listdir(csv_directory) if i.endswith( suffix )]
full_data = []
for i in range(len(csv_files)):
    data = pd.read_csv(csv_directory +csv_files[i], sep=';', index_col = 0)
    full_data.append(data)
    
full_data = pd.concat(full_data, axis=0)


#%%
# Cleaning



# Clean rows with no corresponding image
for sample in full_data.iloc[:,1]:
    try:
        img = Image.open("C:/Users/Ben/Supercase/to_commit/mining-optimizer/data/HQ_digital_contrasted/" + sample)    
    except FileNotFoundError:
        full_data = full_data[full_data.image != sample]
        
# Formating
full_data["used_liter"] = full_data["used_liter"].astype(str)

full_data["used_liter"] = np.where(full_data['used_liter'].apply(len) == 2, "x" + full_data['used_liter'], full_data['used_liter'])
full_data["used_liter"] = np.where(full_data['used_liter'].apply(len) == 3, "x" + full_data['used_liter'], full_data['used_liter'])
        
X =  full_data.iloc[:,1]
y =  full_data.iloc[:,0]

#%%
# Mutliple

# Séparation training test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

temp = []
for sample in X_train:
    try:
        img = Image.open("C:/Users/Ben/Supercase/to_commit/mining-optimizer/data/HQ_digital_contrasted/" + sample)    
    except FileNotFoundError:
        y_train.drop(full_data.loc[full_data["image"] == "46fa7d1747356d68ecae9b59cd6ae8086fd6123f.jpg", "used_liter"])
    img = np.array(img)
    img.resize((100,256))
    img = img.reshape((img.shape[0],img.shape[1],1))
    temp.append(img)
    
X_train = np.asarray(temp)

temp = []
for sample in X_test:
    try:
        img = Image.open("C:/Users/Ben/Supercase/to_commit/mining-optimizer/data/HQ_digital/" + sample)    
    except FileNotFoundError:
        continue
    img = np.array(img)
    img.resize((100,256))
    img = img.reshape((img.shape[0],img.shape[1],1))
    temp.append(img)
    
X_test = np.asarray(temp)

def replacex(x,i):
    if x[i] == "x":
        return 10
    else:
        return int(x[i])

y_train_vect = [y_train.apply(lambda x: replacex(x,0)),y_train.apply(lambda x: replacex(x,1)), y_train.apply(lambda x: replacex(x,2)), y_train.apply(lambda x: replacex(x,3))]
y_test_vect = [y_test.apply(lambda x: replacex(x,0)),y_test.apply(lambda x: replacex(x,1)), y_test.apply(lambda x: replacex(x,2)), y_test.apply(lambda x: replacex(x,3))]

#%%
# Model definition

model_input = Input((100,256,1))

x = Conv2D(32, (3, 3), padding='same', name='conv2d_hidden_1', kernel_regularizer=regularizers.l2(0.01))(model_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_1')(x)
x = Dropout(0.30)(x)

x = Conv2D(64, (3, 3), padding='same', name='conv2d_hidden_2', kernel_regularizer=regularizers.l2(0.01))(x)
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

x = Dense(256, activation ='relu', kernel_regularizer=regularizers.l2(0.01))(x)

digit1 = (Dense(output_dim =11,activation = 'softmax', name='digit_1'))(x)
digit2 = (Dense(output_dim =11,activation = 'softmax', name='digit_2'))(x)
digit3 = (Dense(output_dim =11,activation = 'softmax', name='digit_3'))(x)
digit4 = (Dense(output_dim =11,activation = 'softmax', name='digit_4'))(x)

outputs = [digit1, digit2, digit3, digit4]

model = keras.models.Model(input = model_input , output = outputs)
model._make_predict_function()

#%%
# train
lr = 1e-3
epochs = 50

optimizer = Adam(lr=lr, decay=lr/10)
model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics = ['accuracy'])
keras.backend.get_session().run(tf.initialize_all_variables())
history = model.fit(X_train, y_train_vect, batch_size= 50, nb_epoch=epochs, verbose=1, validation_data=(X_test, y_test_vect))

#%%
# Plot loss

for i in range(1,5):
            plt.figure(figsize=[8,6])
            plt.plot(history.history['digit_%i_loss' %i],'r',linewidth=0.5)
            plt.plot(history.history['val_digit_%i_loss' %i],'b',linewidth=0.5)
            plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
            plt.xlabel('Epochs ',fontsize=16)
            plt.ylabel('Loss',fontsize=16)
            plt.title('Loss Curves Digit %i' %i,fontsize=16)
            plt.show()


#%%
# Plot acc
            
for i in range(1,5):
            plt.figure(figsize=[8,6])
            plt.plot(history.history['digit_%i_acc' %i],'r',linewidth=0.5)
            plt.plot(history.history['val_digit_%i_acc' %i],'b',linewidth=0.5)
            plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
            plt.xlabel('Epochs ',fontsize=16)
            plt.ylabel('Accuracy',fontsize=16)
            plt.title('Accuracy Curves Digit %i' %i,fontsize=16)
            plt.show()

#%%
# Predict
            
y_pred = model.predict(X_test)
correct_preds = 0
        
for i in range(X_test.shape[0]):
    pred_list_i = [np.argmax(pred[i]) for pred in y_pred]
    val_list_i  = [y_test_vect[e].values[i] for e in range(4)]
    correct_preds += sum([val_list_i[e] == pred_list_i[e] for e in range(4)])
print('exact accuracy', correct_preds *25 / X_test.shape[0])
    
#mse = 0 
#diff = []
#for i in range(X_test.shape[0]):
#        pred_list_i = [np.argmax(pred[i]) for pred in y_pred]
#        pred_number = 1000* pred_list_i[0] + 100* pred_list_i[1] + 10 * pred_list_i[2] + 1* pred_list_i[3]
#        val_list_i  = int(y_test.values[i])
#        val_number = 1000* val_list_i[0] + 100*  val_list_i[1] + 10 *  val_list_i[2] + 1*  val_list_i[3]
#        diff.append(val_number - pred_number)
#print('difference label vs. prediction', diff)
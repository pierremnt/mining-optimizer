# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:01:44 2019

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
from keras.utils import plot_model
from keras.callbacks import TensorBoard,EarlyStopping
from Datasets import Dataset_Multi, Dataset_Single
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


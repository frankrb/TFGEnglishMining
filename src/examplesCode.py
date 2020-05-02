# DNNClassifier on CSV input dataset.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from sklearn.model_selection import train_test_split

from packaging import version
from datetime import datetime
from time import time
import pandas as pd
import numpy as np
import os as  os
import keras as keras
import shutil
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import tensorflow as tf
#from tensorflow import keras
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.optimizers import Adamax
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l2
from keras.regularizers import l1
from keras.constraints import unit_norm
from keras.optimizers import SGD
from keras.layers import Dense

neurons = 16
ACTIVATION = 'softmax'

def main():

    train_x = []
    train_y = []

    model = keras.models.Sequential([

        keras.layers.Dense(neurons, input_shape=(148,), activation='relu', name='input'),

        keras.layers.Dense(neurons, activation='relu', name='fc2'),

        keras.layers.Dense(3, activation=ACTIVATION, name='output'),

    ])

    model = keras.model.Sequential
    model.add(Dense(neurons, input_dim=148, activation='relu', name='input_layer'))
    model.add(Dense(neurons, activation='relu', name='hidden_layer'))
    model.add(Dense(3, activation='softmax', name='output_layer'))


    # Compile model

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    model.fit(train_x, train_y, epochs=100)

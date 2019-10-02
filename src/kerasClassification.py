from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from keras import optimizers
import pickle
from time import time

import functools
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow_datasets as tfds
import csv

EPOCS = 100
LEARNING_RATE = 0.0035

def kerasClassification():
    print("**Starting Keras**")

    # load dataset
    dataframe = pd.read_csv('../data/Temporales/train_clean.csv', header=None, skiprows=1)
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    dataset = dataframe.values
    X = dataset[:, 0:148].astype(float)
    Y = dataframe.iloc[:, -1:].values

    print("X: ", X)
    print("Y: ", Y)

    # encode class values as integers
    """ Ejemplo
        Iris-setosa,	Iris-versicolor,	Iris-virginica
        1,		0,			0
        0,		1, 			0
        0, 		0, 			1
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    """

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    encoded_Y = encoder.fit_transform(Y)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = utils.to_categorical(encoded_Y)

    estimator = KerasClassifier(build_fn=baseline_model, batch_size=5, verbose=0)
    estimator.fit(X, Y, epochs=EPOCS)

    # load test dataset
    dataframe_test = pd.read_csv('../data/Temporales/test_clean.csv', header=None, skiprows=1)
    dataset_test = dataframe_test.values
    X_Test = dataset_test[:, 0:148].astype(float)
    Y_Test = dataframe_test.iloc[:, -1:].values

    predictions = estimator.predict(X_Test)
    i = 0
    total = 0
    for pred in predictions:
        print("Predicción instancia, valor real: %.2f , %.2f" % (pred, Y_Test[i]))
        if pred == Y_Test[i]:
            total = total + 1
        i=i+1

    print("Accuracy test: ", total/111, "%")

    # aplicamos kfold cross validation
    kfold = KFold(n_splits=10, shuffle=True)

    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Baseline Model Accuracy KFold: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


# define baseline model
def baseline_model():
    # create model
    # inputs de 148 (hay 148 atributos)
    # capa oculta de 8 nodos
    # capa de salida de 3 nodos (3 clases)
    model = Sequential()
    model.add(Dense(128, input_dim=148, activation='relu'))
    model.add(Dense(128, input_dim=148, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
        kerasClassification()
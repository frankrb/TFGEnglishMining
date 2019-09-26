from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
import random
from sklearn.model_selection import KFold
import pickle
from time import time

import functools
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import csv


def basicClassificationTF():

    print("Executing Basic Classification TF - Supervised Machine Learning")
    print("################################################")

    # Nombre de las clases por niveles 1(advanced), 2(intermediate) y 3(elementary) respectivamente
    class_names = ['advanced','intermediate','elementary']

    # cargamos los datos de train y test

    train_data = pd.read_csv('../data/Temporales/train_clean.csv', header=None, skiprows=1)
    test_data = pd.read_csv('../data/Temporales/test_clean.csv', header=None, skiprows=1)

    train_data = train_data.reindex(np.random.permutation(train_data.index))

    print(train_data)

    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)

    print(train_data.head())
    print(test_data.head())

    # Feature Matrix
    train_data_features = train_data.iloc[:, :-1].values
    test_data_features = test_data.iloc[:,:-1].values

    # Data labels
    train_data_labels = train_data.iloc[:, -1:].values
    test_data_labels = test_data.iloc[:, -1:].values

    print("Shape of Feature Matrix Train:", train_data_features.shape, "\nValues:", train_data_features)
    print("Shape Label Vector Train:", train_data_labels.shape, "\nValues:", train_data_labels)

    print("Shape of Feature Matrix Test:", test_data_features.shape, "\nValues:", train_data_features)
    print("Shape Label Vector Test:", test_data_labels.shape, "\nValues:", train_data_labels)

    # creamos el modelo
    # input_shape=(148,0)-> porque las features es un array de 1 dimensión con 148 atributos
    # 128 nodos (neuronas) es el primer layer
    # 3 nodos de softmax-layer -> cada nodo contiene un valor que indica la probabilidad de que
    # la imagen actual pertenezca a a una de las 3 clases

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(148,)),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data_features,train_data_labels,epochs=200)

    test_loss, test_acc = model.evaluate(test_data_features, test_data_labels)

    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    predictions = model.predict(test_data_features)
    print('Predicción del modelo: ', np.argmax(predictions[0]))
    print('Valor real: ',test_data_features[0])


if __name__ == '__main__':
        basicClassificationTF()
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

    #####################
    # grid search
    # learn_rate, neurons_per_layer = grid_search_hyperparameters(train_x, train_y)
    # buscando los mejores parámetros para nuestro modelo
    #####################

    # model.fit(train_x, train_y, epochs=EPOCHS, callbacks=[tensorboard_callback, early_stopping_callback], validation_data=(dev_x, dev_y))
    # Para poder observar el entrenamiento de los datos en TB: tensorboard --logdir=C:\Users\frank\PycharmProjects\TFGEnglishMining\src\logs\
    """
    # Test on unseen data
    results = best_model.evaluate(test_x, test_y)
    print('Final test set loss: {0}, Final test set accuracy: {1}, Learning rate: {2}, Neurons per layer: {3}'.format(results[0], results[1], best_learning_rate, best_neurons_per_layer))
    print('Final test set accuracy: {:4f}'.format(results[1]))


    predictions = best_model.predict_classes(test_x)
    i = 0
    total = 0

    #test_y = dataframe_test.iloc[:, -1:].values

    for pred in predictions:
        # print("Predicción instancia, valor real: %.2f , %.2f" % (pred, test_y[i]))
        print("Predicción instancia: {0}, valor real: {1}".format(CLASS_VALUE[int(pred)], CLASS_VALUE[int(test_y[i])]))
        if pred == test_y[i]:
            total = total + 1
        i = i + 1

    print("Accuracy test: ", (total / 111)*100, "%")
    """

def create_model(learn_rate=0.01, neurons=1):
    model = keras.models.Sequential([
        keras.layers.Dense(neurons, input_shape=(148,), activation='relu', name='fc1'),
        keras.layers.Dense(neurons, activation='relu', name='fc2'),
        keras.layers.Dense(3, activation=ACTIVATION, name='output'),
    ])
    # Compile model
    optimizer = Adam(lr=learn_rate)
    model.compile(loss=LOSS, optimizer=optimizer, metrics=['accuracy'])
    return model

def grid_search_hyperparameters(train_x, train_y):
    model = KerasClassifier(build_fn=create_model, epochs=EPOCHS, verbose=0)
    # define the grid search parameters
    learn_rate = [0.0035, 0.0030, 0.0025, 0.0020, 0.0015, 0.001]
    neurons = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    param_grid = dict(learn_rate=learn_rate, neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(train_x, train_y)

    # summarize results
    print("####### Resultados GridSearchCV ##########")
    print("Mejor: %f usando %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("####### Fin Resultados GridSearchCV ##########")
    #####################

    print(float(grid_result.best_params_['learn_rate']))

    learn_rate = float(grid_result.best_params_['learn_rate'])
    neurons_per_layer = int(grid_result.best_params_['neurons'])

    return learn_rate, neurons_per_layer

if __name__ == '__main__':
        basicClassificationTF()
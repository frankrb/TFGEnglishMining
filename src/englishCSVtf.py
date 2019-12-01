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

"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
"""
TRAINING = "../data/Temporales/train_clean.csv"
TRAIN = "../data/Train/train.csv"
DEV = "../data/Dev/dev.csv"
TEST = "../data/Test/test.csv"
LEARN_RATE = 0.0015 # es el mejor ratio de aprendizaje para este caso
BATCH_SIZE = 32 # en el caso de modelos secuenciales es mejor no incluir el batch size
CLASS_VALUE = ['Advanced', 'Intermediate', 'Elementary'] # advanced = 0, intermediate = 1, elementary = 2
EPOCHS = 100 # número de entrenamiento del modelo
NEURONS_PER_LAYER = 32 # número de nodos por capa(4-16-64-512)
VERBOSITY = 0 # es un parámetro que nos sirve para la forma en que se muestran las epochs
LOSS = 'sparse_categorical_crossentropy' # esto es debido a que las clases están dadas al modelo como un array de dimensión 1 ej.:[1, 2, 0, 0, 1,...]
ACTIVATION = 'sigmoid' #''softmax' # debido a que es una clasificación multiclass (sigmoid en caso de binarias)
DEV_SIZE = 0.01 # porcentaje del dev
LEARN_RATES = [0.001, 0.0015, 0.0020]


def main():
    """
        A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
    """
    create_train_test(20, TRAINING)
    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
        "This notebook requires TensorFlow 2.0 or above."

    # reordenar las columnas del TEST como el TRAIN y lo guardamos
    """
    train_example = pd.read_csv(TRAIN)
    columns = train_example.columns.values.tolist()
    dataframe = pd.read_csv('../data/Temporales/test_clean.csv')
    #dataframe.reindex(columns, axis=1, )
    dataframe = dataframe[columns]
    print(columns)
    dataframe.to_csv('../data/Test/test.csv', header=True, index=None)"""
    #####################

    #####################
    # load train dataset
    dataframe = pd.read_csv(TRAIN, header=None, skiprows=1)
    # dataframe = dataframe.reindex(np.random.permutation(dataframe.index)) //no debemos hacer shuffle siempre que entrenamos o nos darán valores siempre distintos
    dataset = dataframe.values
    # print(dataset)
    # exit(1)
    train_x = dataset[:, 0:148].astype(float)
    # normalize the data attributes
    train_x = preprocessing.normalize(train_x)
    train_y = dataframe.iloc[:, -1:].values

    #train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=DEV_SIZE)

    # load dev dataset
    dataframe = pd.read_csv(DEV, header=None, skiprows=1)
    # dataframe = dataframe.reindex(np.random.permutation(dataframe.index)) //no debemos hacer shuffle siempre que entrenamos o nos darán valores siempre distintos
    dataset = dataframe.values
    # print(dataset)
    # exit(1)
    dev_x = dataset[:, 0:148].astype(float)
    # normalize the data attributes
    dev_x = preprocessing.normalize(dev_x)
    dev_y = dataframe.iloc[:, -1:].values

    # load test dataset
    dataframe = pd.read_csv(TEST, header=None, skiprows=1)
    dataset = dataframe.values
    test_x = dataset[:, 0:148].astype(float)
    # normalize the data attributes
    test_x = preprocessing.normalize(test_x)
    test_y = dataframe.iloc[:, -1:].values
    #####################

    tensorboard_callback = TensorBoard(log_dir="logs\{}".format(time()))
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    # logdir = "logs\scalars\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    #####################
    # grid search
    # learn_rate, neurons_per_layer = grid_search_hyperparameters(train_x, train_y)
    # buscando los mejores parámetros para nuestro modelo
    best_learning_rate, best_neurons_per_layer, best_model = get_best_hyperparameters(train_x, train_y, dev_x, dev_y)
    """
    # Train the model
    model = keras.models.Sequential([
        keras.layers.Dense(best_neurons_per_layer, input_shape=(148,), activation='relu', kernel_regularizer=l2(0.01), name='fc1'),
        keras.layers.Dense(best_neurons_per_layer, activation='relu', kernel_regularizer=l2(0.01), name='fc2'),
        keras.layers.Dense(3, activation=ACTIVATION, name='output'),
    ])

    # Compile model
    optimizer = Adam(lr=best_learning_rate)
    model.compile(loss=LOSS, optimizer=optimizer, metrics=['accuracy'])
    """

    # model.fit(train_x, train_y, epochs=EPOCHS, callbacks=[tensorboard_callback, early_stopping_callback], validation_data=(dev_x, dev_y))
    # Para poder observar el entrenamiento de los datos en TB: tensorboard --logdir=C:\Users\frank\PycharmProjects\TFGEnglishMining\src\logs\
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

def create_train_test(test_percent, csv_file):
    # esta función nos permite crear un train y un dev equilibrado
    # de este modo se obtiene un train-dev-test con equilibrio
    train = 0
    test = 0
    data = pd.read_csv(csv_file)
    # print(data.groupby('level').count())
    num_rows_0 = data.groupby('level').count().iloc[0, -1]
    num_rows_1 = data.groupby('level').count().iloc[1, -1]
    num_rows_2 = data.groupby('level').count().iloc[2, -1]

    num_rows_0 = num_rows_0*test_percent//100
    num_rows_1 = num_rows_1*test_percent//100
    num_rows_2 = num_rows_2*test_percent//100
    # creamos un csv train y dev
    data_train = pd.read_csv(csv_file)
    data_test = pd.read_csv(csv_file)
    i = 0
    data_train_drop = []
    data_test_drop = []
    total_rows = len(data.index)
    print(total_rows)
    print(data.iloc[455, -1])
    while i < total_rows:
        # caso de clase 0
        if(data.iloc[i, -1] == 0) and (num_rows_0 != 0):
            data_train_drop.append(i)
            num_rows_0 -= 1
        elif(data.iloc[i, -1] == 0) and (num_rows_0 == 0):
            data_test_drop.append(i)
        # caso de clase 1
        if (data.iloc[i, -1] == 1) and (num_rows_1 != 0):
            data_train_drop.append(i)
            num_rows_1 -= 1
        elif (data.iloc[i, -1] == 1) and (num_rows_1 == 0):
            data_test_drop.append(i)
        # caso de clase 2
        if (data.iloc[i, -1] == 2) and (num_rows_2 != 0):
            data_train_drop.append(i)
            num_rows_2 -= 1
        elif (data.iloc[i, -1] == 2) and (num_rows_2 == 0):
            data_test_drop.append(i)
        i += 1
    print(data_train_drop)
    print(data_test_drop)
    data_train.drop(data_train_drop, axis=0, inplace=True)
    data_test.drop(data_test_drop, axis=0, inplace=True)

    data_test.to_csv('../data/Dev/dev.csv', header=True, index=None)
    data_train.to_csv('../data/Train/train.csv', header=True, index=None)

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

def get_best_hyperparameters(train_x, train_y, dev_x, dev_y):
    # eliminamos los datos de la carpeta de logs de callbacks
    # shutil.rmtree('trainingLogs/')
    # definimos los parámetros a entrenar con el modelo
    learn_rate = [0.0010, 0.00125, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035]
    # el número de neuronas por capa debe de ser bajo para no producir overfitting
    # si el número de neuronas es alto la diferencia entre el acc del train y del dev son muy grandes
    # por lo que el acc del modelo no será el real
    # tener más neuronas por capa hará el modelo más complejo y a su vez más inestable(muchas subidas y bajadas del acc)
    # neurons = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    # entrenameros con neuronas en el rango 4-64 para obtener modelos más estables
    neurons = [8, 16, 32, 48, 64]
    best_learn_rate = learn_rate[0]
    best_neurons_per_layer = neurons[0]
    best_accuracy = 0
    best_model = None
    for lr in learn_rate:
        for nrs in neurons:
            directory_name = str(lr),"_",str(nrs)
            # haremos un early stopping si en las siguientes 3 epocs del modelo no se produce una mejora
            early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            training_tensorboard_callback = TensorBoard(log_dir="trainingLogs\{}".format(directory_name))
            # añadiremos una capa dropout para evitar overfitting
            model = keras.models.Sequential([
                keras.layers.Dense(nrs, input_shape=(148,), activation='relu', name='input', activity_regularizer=l1(0.001)),
                #keras.layers.Dense(nrs, activation='relu', name='fc0'),
                keras.layers.Dropout(0.2, name='do1'),
                keras.layers.Dense(nrs, activation='relu', name='fc1'),
                keras.layers.Dropout(0.2, name='do2'),
                keras.layers.Dense(3, activation='softmax', name='output'),
            ])
            optimizer = Adam(lr=lr)
            #optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss=LOSS, optimizer=optimizer, metrics=['accuracy'])
            model.fit(train_x, train_y, epochs=EPOCHS, verbose=0, callbacks=[training_tensorboard_callback, early_stopping_callback], validation_data=(dev_x, dev_y))
            # Para poder observar el entrenamiento de los datos en TB: tensorboard --logdir=C:\Users\frank\PycharmProjects\TFGEnglishMining\src\trainingLogs\
            # Evaluamos el modelo obtenido
            results = model.evaluate(dev_x, dev_y)
            print('Dev set loss: {0}, Dev set accuracy: {1}, Learning rate: {2}, Neurons: {3}'.format(results[0], results[1], lr, nrs))
            # comprobamos si hemos obtenido un mejor accuracy
            if best_accuracy < results[1]:
                best_accuracy = results[1]
                best_learn_rate = lr
                best_neurons_per_layer = nrs
                best_model = model

    print('Final best Acc: {0}, Final best learning_rate {1}, Final best neurons per layer {2}'.format(best_accuracy, best_learn_rate, best_neurons_per_layer))

    return best_learn_rate, best_neurons_per_layer, best_model

if __name__ == "__main__":
    main()
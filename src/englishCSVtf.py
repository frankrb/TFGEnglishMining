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
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import tensorflow as tf
#from tensorflow import keras
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.optimizers import SGD

"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
"""
TRAINING = "../data/Temporales/train_clean.csv"
TEST = "../data/Temporales/test_clean.csv"
LEARN_RATE = 0.0015 # es el mejor ratio de aprendizaje para este caso
BATCH_SIZE = 320 # en el caso de modelos secuenciales es mejor no incluir el batch size
CLASS_VALUE = ['Advanced', 'Intermediate', 'Elementary'] # advanced = 0, intermediate = 1, elementary = 2
EPOCHS = 125 # número de entrenamiento del modelo
L2_VALUE = 0.001
NEURONS_PER_LAYER = 32 # número de nodos por capa(4-16-64-512)
VERBOSITY = 0 # es un parámetro que nos sirve para la forma en que se muestran las epochs
LOSS = 'sparse_categorical_crossentropy' # esto es debido a que las clases están dadas al modelo como un array de dimensión 1 ej.:[1, 2, 0, 0, 1,...]
ACTIVATION = 'softmax' # debido a que es una clasificación multiclass (sigmoid en caso de binarias)
DEV_SIZE = 0.01 # porcentaje del dev
LEARN_RATES = [0.001,0.0015,0.0020]


def main():
    """
        A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
    """

    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
        "This notebook requires TensorFlow 2.0 or above."

    #####################
    # load train dataset
    dataframe = pd.read_csv(TRAINING, header=None, skiprows=1)
    # dataframe = dataframe.reindex(np.random.permutation(dataframe.index)) //no debemos hacer shuffle siempre que entrenamos o nos darán valores siempre distintos
    dataset = dataframe.values
    train_x = dataset[:, 0:148].astype(float)
    # normalize the data attributes
    train_x = preprocessing.normalize(train_x)
    train_y = dataframe.iloc[:, -1:].values

    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=DEV_SIZE)

    # load test dataset
    dataframe = pd.read_csv(TEST, header=None, skiprows=1)
    dataset = dataframe.values
    test_x = dataset[:, 0:148].astype(float)
    # normalize the data attributes
    test_x = preprocessing.normalize(test_x)
    test_y = dataframe.iloc[:, -1:].values
    #####################

    tensorboard_callback = TensorBoard(log_dir="logs\{}".format(time()))

    # logdir = "logs\scalars\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)


    # Adam optimizer with learning rate of 0.001
    #optimizer = Adam(lr=0.0035)
    #model = create_model()
    #model.compile(tf.keras.optimizers.Adam(lr=0.00035), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    """"
    #####################
    model = KerasClassifier(build_fn=create_model, epochs=EPOCHS, verbose=0)
    # define the grid search parameters
    learn_rate = [0.0035, 0.0030, 0.0025, 0.0020, 0.0015]
    neurons = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
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
    """

    # Train the model
    model = keras.models.Sequential([
        keras.layers.Dense(NEURONS_PER_LAYER, input_shape=(148,), activation='relu', kernel_regularizer=keras.regularizers.l2(L2_VALUE), name='fc1'),
        keras.layers.Dense(NEURONS_PER_LAYER, activation='relu', kernel_regularizer=keras.regularizers.l2(L2_VALUE), name='fc2'),
        keras.layers.Dense(3, activation=ACTIVATION, name='output'),
    ])

    # Compile model
    # optimizer = Adam(lr=float(grid_result.best_params_['learn_rate']))
    optimizer = Adam(lr=LEARN_RATES[2])
    model.compile(loss=LOSS, optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tensorboard_callback], validation_data=(test_x, test_y))
    # Para poder observar el entrenamiento de los datos en TB: tensorboard --logdir=C:\Users\frank\PycharmProjects\TFGEnglishMining\src\logs\
    # Test on unseen data
    results = model.evaluate(test_x, test_y)
    print('Final test set loss: {0}, Final test set accuracy: {1}, Learning rate: {2}'.format(results[0], results[1], LEARN_RATES[2]))
    print('Final test set accuracy: {:4f}'.format(results[1]))


    predictions = model.predict_classes(test_x)
    i = 0
    total = 0

    #test_y = dataframe_test.iloc[:, -1:].values

    for pred in predictions:
        # print("Predicción instancia, valor real: %.2f , %.2f" % (pred, test_y[i]))
        print("Predicción instancia: {0}, valor real: {1}".format(CLASS_VALUE[int(pred)], CLASS_VALUE[int(test_y[i])]))
        if pred == test_y[i]:
            total = total + 1
        i = i + 1

    print("Accuracy test: ", total / 111, "%")

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

if __name__ == "__main__":
    main()
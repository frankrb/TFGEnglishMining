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
LEARN_RATE = 0.01


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
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    dataset = dataframe.values
    train_x = dataset[:, 0:148].astype(float)
    # normalize the data attributes
    train_x = preprocessing.normalize(train_x)
    train_y = dataframe.iloc[:, -1:].values

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.25)

    #####################

    tensorboard_callback = TensorBoard(log_dir="logs\{}".format(time()))

    # logdir = "logs\scalars\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)


    # Adam optimizer with learning rate of 0.001
    #optimizer = Adam(lr=0.0035)
    #model = create_model()
    #model.compile(tf.keras.optimizers.Adam(lr=0.00035), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    #####################
    model = KerasClassifier(build_fn=create_model, epochs=40, batch_size=10, verbose=0)
    # define the grid search parameters
    learn_rate = [0.0035, 0.0030, 0.0025, 0.0020, 0.0015, 0.0010, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004]
    param_grid = dict(learn_rate=learn_rate)
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

    # Train the model
    model = keras.models.Sequential([
        keras.layers.Dense(16, input_shape=(148,), activation='relu', name='fc1'),
        keras.layers.Dense(16, activation='relu', name='fc2'),
        keras.layers.Dense(3, activation='softmax', name='output'),
    ])

    # Compile model
    optimizer = Adam(lr=float(grid_result.best_params_['learn_rate']))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=40, callbacks=[tensorboard_callback])
    # Para poder observar el entrenamiento de los datos en TB: tensorboard --logdir=C:\Users\frank\PycharmProjects\TFGEnglishMining\src\logs\

    # Test on unseen data

    results = model.evaluate(test_x, test_y)

    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))

    predictions = model.predict_classes(test_x)
    i = 0
    total = 0

    #test_y = dataframe_test.iloc[:, -1:].values

    for pred in predictions:
        print("Predicción instancia, valor real: %.2f , %.2f" % (pred, test_y[i]))
        if pred == test_y[i]:
            total = total + 1
        i = i + 1

    print("Accuracy test: ", total / 111, "%")

def create_model(learn_rate=0.01):
    model = keras.models.Sequential([
        keras.layers.Dense(16, input_shape=(148,), activation='relu', name='fc1'),
        keras.layers.Dense(16, activation='relu', name='fc2'),
        keras.layers.Dense(3, activation='softmax', name='output'),
    ])
    # Compile model
    optimizer = Adam(lr=learn_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    main()
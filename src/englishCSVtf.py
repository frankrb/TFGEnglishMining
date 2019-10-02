# DNNClassifier on CSV input dataset.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

TRAINING = "../data/Temporales/train_clean.csv"
TEST = "../data/Temporales/test_clean.csv"


def main():
    """
        A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
    """

    #####################
    # load train dataset
    dataframe = pd.read_csv(TRAINING, header=None, skiprows=1)
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    dataset = dataframe.values
    train_x = dataset[:, 0:148].astype(float)
    # normalize the data attributes
    train_x = preprocessing.normalize(train_x)
    train_y = dataframe.iloc[:, -1:].values
    #####################

    #####################
    # load test dataset
    dataframe_test = pd.read_csv(TEST, header=None, skiprows=1)
    dataset_test = dataframe_test.values
    test_x = dataset_test[:, 0:148].astype(float)
    # normalize the data attributes
    test_x = preprocessing.normalize(test_x)
    test_y = dataframe_test.iloc[:, -1:].values
    #####################

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    # Split the data for training and testing
    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)


    # Build the model

    model = Sequential()

    model.add(Dense(64, input_shape=(148,), activation='relu', name='fc1'))
    model.add(Dense(64, activation='relu', name='fc2'))
    model.add(Dense(3, activation='softmax', name='output'))

    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=0.0035)
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('Neural Network Model Summary: ')
    print(model.summary())

    # Train the model
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=8)

    # Test on unseen data

    results = model.evaluate(test_x, test_y)

    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))

    predictions = model.predict_classes(test_x)
    i = 0
    total = 0

    test_y = dataframe_test.iloc[:, -1:].values

    for pred in predictions:
        print("Predicción instancia, valor real: %.2f , %.2f" % (pred, test_y[i]))
        if pred == test_y[i]:
            total = total + 1
        i = i + 1

    print("Accuracy test: ", total / 111, "%")

if __name__ == "__main__":
    main()
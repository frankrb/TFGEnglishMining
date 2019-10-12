
# DNNClassifier on CSV input dataset.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd

from sklearn import preprocessing


# Data sets
IRIS_TRAINING = "../data/NewData/iris_training.csv"
IRIS_TEST = "../data/NewData/iris_test.csv"
EPOCH = 2000
"""
def main():
    # Load datasets.

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    # Build 3 layer DNN
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[5, 10, 5],
                                                n_classes=3)

    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)

        return x, y

    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=EPOCH)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)

        return x, y

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify new flower
    def new_samples():
        return np.array([[6.4, 2.7, 5.6, 2.1]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples))

    print("Predicted class: {}\n".format(predictions))

"""

def main():
    """
        A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
    """

    #####################
    # load train dataset
    dataframe = pd.read_csv('../data/NewData/iris_training.csv', header=None, skiprows=1)
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    dataset = dataframe.values
    train_x = dataset[:, 0:4].astype(float)
    # normalize the data attributes
    train_x = preprocessing.normalize(train_x)
    train_y = dataframe.iloc[:, -1:].values
    print(train_x)
    print(train_y)
    #####################

    #####################
    # load test dataset
    dataframe_test = pd.read_csv('../data/NewData/iris_test.csv', header=None, skiprows=1)
    dataset_test = dataframe_test.values
    test_x = dataset_test[:, 0:4].astype(float)
    # normalize the data attributes
    test_x = preprocessing.normalize(test_x)
    test_y = dataframe_test.iloc[:, -1:].values
    #####################

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    #train_y = encoder.fit_transform(train_y)
    #test_y = encoder.fit_transform(test_y)

    # Split the data for training and testing
    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)


    # Build the model

    model = Sequential()

    model.add(Dense(15, input_shape=(4,), activation='relu', name='fc1'))
    model.add(Dense(15, activation='relu', name='fc2'))
    model.add(Dense(3, activation='softmax', name='output'))

    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=0.0035)
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('Neural Network Model Summary: ')
    print(model.summary())

    # Train the model
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=10)

    # Test on unseen data

    results = model.evaluate(test_x, test_y)

    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))

    predictions = model.predict_classes(test_x)
    i = 0
    total = 0

    print(predictions)

    test_y = dataframe_test.iloc[:, -1:].values

    for pred in predictions:
        print("Predicción instancia, valor real: %.2f , %.2f" % (pred, test_y[i]))
        if pred == test_y[i]:
            total = total + 1
        i = i + 1

    print("Accuracy test: ", total / 31, "%")

if __name__ == "__main__":
    main()
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
from sklearn.model_selection import train_test_split
import functools
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_datasets as tfds
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def logisticRegresion():

    print("Executing logistic regression with TensorFlow - Supervised Machine Learning")
    print("################################################")

    data = pd.read_csv('../data/Temporales/train_clean.csv',header=None, skiprows=1)
    data = data.reindex(np.random.permutation(data.index))

    # data.describe()

    print("Data Shape:", data.shape)

    print(data.head());

    # normalizamos los datos


    # Feature Matrix
    x_orig = data.iloc[:, :-1].values
    x = preprocessing.normalize(x_orig)

    # Data labels
    y_orig = data.iloc[:, -1:].values
    y = y_orig

    ##############
    data_test = pd.read_csv('../data/Temporales/test_clean.csv', header=None, skiprows=1)
    data_test = data_test.reindex(np.random.permutation(data_test.index))

    # data.describe()

    print("Data Shape:", data_test.shape)

    print(data_test.head());

    # normalizamos los datos

    # Feature Matrix
    X_test = data_test.iloc[:, :-1].values
    X_test = preprocessing.normalize(X_test)

    # Data labels
    Y_test = data_test.iloc[:, -1:].values
    #############

    print("Train features values: ", x)
    print("Train labels values: ", y)

    alpha, epochs = 0.0015, 10
    m, n = x.shape
    print('m =', m)
    print('n =', n)
    print('Learning Rate =', alpha)
    print('Number of Epochs =', epochs)
    # exit(1)
    # There are n columns in the feature matrix
    # after One Hot Encoding.
    X = tf.placeholder(tf.float32, [None, n])

    # Since this is a binary classification problem,
    # Y can take only 3 values(0-1-2).
    # Hay que ponerle exactamente el número de valores posibles que toma la clase en la muestra, en este caso 3
    Y = tf.placeholder(tf.float32, [None, 1])

    # Trainable Variable Weights
    W = tf.Variable(tf.zeros([n, 1]))

    # Trainable Variable Bias
    b = tf.Variable(tf.zeros([1]))

    # Hypothesis
    Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))

    # Sigmoid Cross Entropy Cost Function
    cost = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=Y_hat, labels=Y)

    # Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=alpha).minimize(cost)

    # Global Variables Initializer
    init = tf.global_variables_initializer()

    # Starting the Tensorflow Session
    with tf.Session() as sess:

        # Initializing the Variables

        sess.run(init)

        # Lists for storing the changing Cost and Accuracy in every Epoch
        cost_history, accuracy_history = [], []

        # Iterating through all the epochs
        for epoch in range(epochs):
            cost_per_epoch = 0

            # Running the Optimizer
            sess.run(optimizer, feed_dict={X: x, Y: y})

            # Calculating cost on current Epoch
            c = sess.run(cost, feed_dict={X: x, Y: y})

            # Calculating accuracy on current Epoch
            correct_prediction = tf.equal(tf.argmax(Y_hat, 1),
                                          tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                              tf.float32))

            # Storing Cost and Accuracy to the history
            cost_history.append(sum(sum(c)))
            accuracy_history.append(accuracy.eval({X: x, Y: y}) * 100)

            # Displaying result on current Epoch
            if epoch % 100 == 0 and epoch != 0:
                print("Epoch " + str(epoch) + " Cost: "
                      + str(cost_history[-1]))

        Weight = sess.run(W)  # Optimized Weight
        Bias = sess.run(b)  # Optimized Bias

        # Save the model (.ckpt)
        saver = tf.train.Saver()
        saver.save(sess, '../data/Modelos/model.ckpt')

        # Final Accuracy
        correct_prediction = tf.equal(tf.argmax(Y_hat, 1),
                                      tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                          tf.float32))

        train_accuracy = accuracy.eval({X: x, Y: y})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("\nAccuracy:", accuracy_history[-1], "%")
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)


    # Let’s plot the change of cost over the epochs.

    plt.plot(list(range(epochs)), cost_history)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Decrease in Cost with Epochs')
    plt.show()

    # Plot the change of accuracy over the epochs.

    plt.plot(list(range(epochs)), accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Increase in Accuracy with Epochs')
    plt.show()

    # Ahora utilizaremos el modelo para predecir
    #testModel()

def testModel():
    saver = tf.train.Saver()
    with tf.Session() as sess:
            saver.restore(sess, '../data/Modelos/model.ckpt')
            print("Model restored.")
            test_accuracy = sess.run
            print("Test Accuracy = {:.3f}".format(test_accuracy[0] * 100))

    return


if __name__ == '__main__':
    logisticRegresion()
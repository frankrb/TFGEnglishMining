"""
usage: englishCSVtf.py [-h] -trdp TRAIN_DATA_PATH -tedp TEST_DATA_PATH -of
                       OUTPUT_FOLDER -ep ENTRENAR_PREDECIR
                       [-mv METODO_VALIDACION] [-sa SELECCION_ATRIBUTOS]
                       [-tsa TIPO_SELECCION_ATRIBUTOS] [-na NUM_ATRIBUTOS]
                       [-mp SOURCE_MODEL_PATH]

optional arguments:
  -h, --help            Muestra el mensaje de ayuda.
  -trdp TRAIN_DATA_PATH, --train_data_path TRAIN_DATA_PATH
                        ruta del archivo csv de entrenamiento.
  -tedp TEST_DATA_PATH, --test_data_path TEST_DATA_PATH
                        ruta del archivo csv de testeo.
  -of OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        ruta del directorio en el que guardar los resultados.
  -ep ENTRENAR_PREDECIR, --entrenar_predecir ENTRENAR_PREDECIR
                        atributo que nos dice si hay que entrenar el
                        modelo(train) o realizar una predicción(predict).
  -mv METODO_VALIDACION, --metodo_validacion METODO_VALIDACION
                        atributo para seleccionar el método de validación:
                        HoldOut o KFold.
  -sa SELECCION_ATRIBUTOS, --seleccion_atributos SELECCION_ATRIBUTOS
                        realizar selección de atributos: True o False.
  -tsa TIPO_SELECCION_ATRIBUTOS, --tipo_seleccion_atributos TIPO_SELECCION_ATRIBUTOS
                        atributo para especificar la selección de atributos:
                        chi2, f_classif o mutual_info_classif.
  -na NUM_ATRIBUTOS, --num_atributos NUM_ATRIBUTOS
                        número de atributos para la selección.
  -mp SOURCE_MODEL_PATH, --source_model_path SOURCE_MODEL_PATH
                        path del modelo entrenado.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from packaging import version
import pandas as pd
import numpy as np
import keras as keras
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l1
import preprocessing as preprocess
from keras.models import load_model
from argparse import ArgumentParser
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
"""
TRAINING = "../data/Temporales/train_clean.csv"
TRAIN = "../data/Train/train.csv"
DEV = "../data/Dev/dev.csv"
TEST = "../data/Test/test.csv"
BATCH_SIZE = 32 # en el caso de modelos secuenciales es mejor no incluir el batch size
CLASS_VALUE = ['Advanced', 'Intermediate', 'Elementary'] # advanced = 0, intermediate = 1, elementary = 2
EPOCHS = 100 # número de entrenamiento del modelo
NEURONS_PER_LAYER = [4, 16, 32, 64] # número de nodos por capa(4-16-64-512)
VERBOSITY = 0 # es un parámetro que nos sirve para la forma en que se muestran las epochs
LOSS = 'sparse_categorical_crossentropy' # esto es debido a que las clases están dadas al modelo como un array de dimensión 1 ej.:[1, 2, 0, 0, 1,...]
ACTIVATION = 'softmax' #''softmax' # debido a que es una clasificación multiclass (sigmoid en caso de binarias)
DEV_SIZE = 0.01 # porcentaje del dev
LEARN_RATES = [0.01, 0.001, 0.00125, 0.0015, 0.002, 0.0025, 0.003, 0.0035]
'''TRAIN_PREDICT = 'predict' # seleccionamos si deseamos entrenar el modelo o simplemente hacer predicciones con uno existente: train o predict
ATRIBUTE_SELECTION_TYPE = '' #seleccionamos los X mejores atributos según chi2, f_classif, mutual_info_classif
ATRIBUTE_SELECTION_NO_ATR = 50
PATH_MODELO_TARGET = '../data/Modelos'
PATH_MODELO_SOURCE = '../data/Modelos/mi_modelo_HoldOut_.h5'
PATH_TRAIN_SOURCE = '../data/Train_Aztertest/train_aztertest.csv'
PATH_TRAIN_TARGET = '../data/Modelos/Train'
PATH_TEST_SOURCE = '../data/Test_Aztertest/test_aztertest.csv'
PATH_TRAIN_SOURCE_TRAINED_MODEL = '../data/Modelos/Train/mi_trainHoldOut_.csv'
PATH_RESULTADOS = ''
'''

def _get_args():
    """Devuelve los argumentos introducidos por la terminal."""
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help',
                        help='Muestra el mensaje de ayuda.')
    parser.add_argument('-trdp', '--train_data_path',
                        type=str,
                        required=True,
                        help='ruta del archivo csv de entrenamiento.')
    parser.add_argument('-tedp', '--test_data_path',
                        type=str,
                        required=True,
                        help='ruta del archivo csv de testeo.')
    parser.add_argument('-of', '--output_folder',
                        type=str,
                        required=True,
                        help='ruta del directorio en el que guardar los resultados.')
    parser.add_argument('-ep', '--entrenar_predecir',
                        type=str,
                        default='train',
                        required=True,
                        help='atributo que nos dice si hay que entrenar el modelo(train) o realizar una predicción(predict).')
    parser.add_argument('-mv', '--metodo_validacion',
                        type=str,
                        default='HoldOut',
                        required=False,
                        help='atributo para seleccionar el método de validación: HoldOut o KFold.')

    parser.add_argument('-sa', '--seleccion_atributos',
                        type=str,
                        default='False',
                        required=False,
                        help='realizar selección de atributos: True o False.')

    parser.add_argument('-tsa', '--tipo_seleccion_atributos',
                        type=str,
                        default='',
                        required=False,
                        help='atributo para especificar la selección de atributos: chi2, f_classif o mutual_info_classif.')
    parser.add_argument('-na', '--num_atributos',
                        type=int,
                        default=50,
                        required=False,
                        help='número de atributos para la selección.')
    parser.add_argument('-mp', '--source_model_path',
                        type=str,
                        required=False,
                        help='path del modelo entrenado.')
    return parser.parse_args()

def main():
    print("Versión de TensorFlow: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
        "Esta aplicación requiere una versión de TensorFlow 2.0 o mayor."

    args = _get_args()

    TRAIN_PREDICT = args.entrenar_predecir
    VALIDATION_METHOD = args.metodo_validacion
    PATH_TEST_SOURCE = args.test_data_path
    PATH_TRAIN_SOURCE = args.train_data_path
    PATH_MODELO_TARGET = args.output_folder
    ATRIBUTE_SELECTION = args.seleccion_atributos
    ATRIBUTE_SELECTION_TYPE = args.tipo_seleccion_atributos
    ATRIBUTE_SELECTION_NO_ATR = args.num_atributos
    PATH_MODELO_SOURCE = args.source_model_path
    PATH_RESULTADOS = args.output_folder

    if TRAIN_PREDICT == 'train':
        # cargamos los datos
        # cambiamos las clases de Train y Test a números enteros
        preprocess.change_class_to_int(PATH_TEST_SOURCE, "../data/DataExamples/test_class_integer.csv")
        preprocess.change_class_to_int(PATH_TRAIN_SOURCE, "../data/DataExamples/train_class_integer.csv")
        # ordenamos las columnas de train y test para que tengan el mismo orden
        preprocess.order_columns("../data/DataExamples/train_class_integer.csv", "../data/DataExamples/test_class_integer.csv",
                      "../data/DataExamples/test.csv")

        # aplicamos un método de validación: KFold o HoldOut
        validatioMethod = VALIDATION_METHOD
        atributeSelection = ATRIBUTE_SELECTION
        atributeSelectionType = ATRIBUTE_SELECTION_TYPE
        numAtributes = ATRIBUTE_SELECTION_NO_ATR

        if validatioMethod == "HoldOut":
            # creamos un train y un dev(20% del train) equilibrado partiendo del train anterior(Método de validación Hold-Out)
            preprocess.create_train_dev(20, "../data/DataExamples/train_class_integer.csv",
                                        "../data/DataExamples/train.csv",
                                        "../data/DataExamples/dev.csv")
            # normalizamos los datos y los cargamos para trabajar con ellos
            train_x, train_y, dev_x, dev_y, test_x, test_y = preprocess.fLoadTrainDevTestNormalizing(
                "../data/DataExamples/train.csv",
                "../data/DataExamples/dev.csv",
                "../data/DataExamples/test.csv")
            if atributeSelection == 'True':
                #seleccionamos los X mejores atributos según chi2, f_classif, mutual_info_classif
                train_x, dev_x, test_x = preprocess.fSelectKbestNormalize(numAtributes, atributeSelectionType, train_x, train_y,
                                                                          dev_x, test_x)
            # obtenemos los mejores hyperparámetros y modelo
            best_learning_rate, best_neurons_per_layer, best_model = get_best_hyperparametersHoldOut(train_x, train_y,
                                                                                                     dev_x, dev_y)
            # guardamos el modelo
            save_model(best_model, PATH_RESULTADOS + '/mi_modelo_'+validatioMethod+'_'+atributeSelectionType+'.h5')
            # guardamos el train para tener su estructura en caso de querer predecir posteriormente con el modelo
            df = pd.read_csv("../data/DataExamples/train.csv")
            df.to_csv(PATH_RESULTADOS + '/mi_train_' + validatioMethod + '_' + atributeSelectionType + '.csv', header=True, index=None)
        elif validatioMethod == "KFold":
            # normalizamos los datos y los cargamos para trabajar con ellos
            train_x, train_y, test_x, test_y = preprocess.fLoadTrainTestNormalizing(
                "../data/DataExamples/train_class_integer.csv",
                "../data/DataExamples/test.csv")
            if atributeSelection == 'True':
                # seleccionamos los X mejores atributos según chi2, f_classif, mutual_info_classif
                train_x, test_x = preprocess.fSelectKbestNormalize1(numAtributes, atributeSelectionType, train_x, train_y,
                                                                           test_x)
            # obtenemos los mejores hyperparámetros y modelo
            best_learning_rate, best_neurons_per_layer, best_model = get_best_hyperparametersKFold(train_x, train_y, 3)
            # guardamos el modelo
            save_model(best_model, PATH_MODELO_TARGET + '/mi_modelo_'+validatioMethod+'_'+atributeSelectionType+'.h5')
            # guardamos el train para tener su estructura en caso de querer predecir posteriormente con el modelo
            df = pd.read_csv("../data/DataExamples/train.csv")
            df.to_csv(PATH_RESULTADOS + '/mi_train_' + validatioMethod + '_' + atributeSelectionType + '.csv', header=True, index=None)
        else:
            exit('Error: Método de validación incorrecto')

        # Test on unseen data
        results = best_model.evaluate(test_x, test_y)
        print(
            'Final test set loss: {0}, Final test set accuracy: {1}, Learning rate: {2}, Neurons per layer: {3}'.format(
                results[0], results[1], best_learning_rate, best_neurons_per_layer))
        print('Final test set accuracy: {:4f}'.format(results[1]))

        predictions = best_model.predict_classes(test_x)
        i = 0
        total = 0

        # test_y = dataframe_test.iloc[:, -1:].values

        for pred in predictions:
            # print("Predicción instancia, valor real: %.2f , %.2f" % (pred, test_y[i]))
            print("Predicción instancia: {0}, valor real: {1}".format(CLASS_VALUE[int(pred)],
                                                                      CLASS_VALUE[int(test_y[i])]))
            if pred == test_y[i]:
                total = total + 1
            i = i + 1
        bind_test_accuracy = (total / i) * 100
        print("Accuracy test: ", bind_test_accuracy, "%")


    elif TRAIN_PREDICT == 'predict':
        # predecimos las clases del test con el modelo proporcionado y la estructura del train con el que se construyó el modelo
        best_model = tf.keras.models.load_model(PATH_MODELO_SOURCE)
        # cambiamos la clase a integer
        preprocess.change_class_to_int(PATH_TEST_SOURCE, "../data/DataExamples/test_class_integer.csv")
        # ordenamos y seleccionamos las columnas que tiene el train usado para entrenar
        preprocess.order_columns(PATH_TRAIN_SOURCE,
                                 "../data/DataExamples/test_class_integer.csv",
                                 "../data/DataExamples/test.csv")
        test_x, test_y = preprocess.fLoadNormalizing("../data/DataExamples/test.csv")

        # Test on unseen data
        results = best_model.evaluate(test_x, test_y)
        print(
            'Final test set loss: {0}, Final test set accuracy: {1}, Learning rate: {2}'.format(
                results[0], results[1], K.eval(best_model.optimizer.lr)))
        #plot_model(best_model, to_file='../data/Modelos/model_plot.png', show_shapes=True, show_layer_names=True)
        print('Final test set accuracy: {:4f}'.format(results[1]))

        predictions = best_model.predict_classes(test_x)
        i = 0
        total = 0

        # test_y = dataframe_test.iloc[:, -1:].values

        for pred in predictions:
            # print("Predicción instancia, valor real: %.2f , %.2f" % (pred, test_y[i]))
            print("Predicción instancia: {0}, valor real: {1}".format(CLASS_VALUE[int(pred)],
                                                                      CLASS_VALUE[int(test_y[i])]))
            if pred == test_y[i]:
                total = total + 1
            i = i + 1

        bind_test_accuracy = (total / i) * 100
        print("Accuracy test: ", bind_test_accuracy, "%")

    else:
        exit('Error: No ha seleccionado ninguna opción predicción o entrenamiento')

    ##Guardamos los resultados en un TXT##
    file = open(PATH_RESULTADOS + '/Resultados.txt', 'a')
    file.write('\n')
    file.write('\n' + '########################')
    file.write('\n' + 'Tipo de ejecución: ' + TRAIN_PREDICT)
    if TRAIN_PREDICT == 'train':
        file.write('\n' + 'Tipo de validación: ' + VALIDATION_METHOD)
        file.write('\n' + 'Selección de atributos: ' + ATRIBUTE_SELECTION)
        if atributeSelection == 'True':
            file.write('\n' + ' Tipo de salección de atributos: ' + ATRIBUTE_SELECTION_TYPE)
            file.write('\n' + 'Nº de atributos seleccionados: ' + ATRIBUTE_SELECTION_NO_ATR)
        file.write(
            '\n' + 'Path del train usado para el entrenamiento: ' + PATH_RESULTADOS + '\mi_train_' + validatioMethod + '_' + atributeSelectionType + '.csv')
        file.write(
            '\n' + 'Path del modelo final: ' + PATH_RESULTADOS + '\mi_modelo_' + validatioMethod + '_' + atributeSelectionType + '.h5')

    elif TRAIN_PREDICT == 'predict':
        file.write(
            '\n' + 'Path del train usado para el entrenamiento: ' + PATH_TRAIN_SOURCE)
        file.write(
            '\n' + 'Path del modelo usado: ' + PATH_MODELO_SOURCE)
    text_accuracy = 'Accuracy test a ciegas: %.3f' % bind_test_accuracy + "%"
    file.write('\n' + text_accuracy)
    file.write('\n' + '########################')
    file.close()
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

def get_best_hyperparametersHoldOut(train_x, train_y, dev_x, dev_y):
    # eliminamos los datos de la carpeta de logs de callbacks
    # shutil.rmtree('trainingLogs/')
    # definimos los parámetros a entrenar con el modelo
    learn_rate = [0.01, 0.001, 0.00125, 0.0015, 0.002, 0.0025, 0.003, 0.0035]
    # el número de neuronas por capa debe de ser bajo para no producir overfitting
    # si el número de neuronas es alto la diferencia entre el acc del train y del dev son muy grandes
    # por lo que el acc del modelo no será el real
    # tener más neuronas por capa hará el modelo más complejo y a su vez más inestable(muchas subidas y bajadas del acc)
    # neurons = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    # entrenameros con neuronas en el rango 4-64 para obtener modelos más estables
    neurons = [4, 16, 32, 64]
    best_learn_rate = learn_rate[0]
    best_neurons_per_layer = neurons[0]
    best_accuracy = 0
    best_model = None
    columns = np.size(train_x,1)
    print(columns)
    for lr in learn_rate:
        for nrs in neurons:
            directory_name = str(lr),"_",str(nrs)
            # haremos un early stopping si en las siguientes 3 epocs del modelo no se produce una mejora
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            training_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="trainingLogs\{}".format(directory_name))
            # añadiremos una capa dropout para evitar overfitting
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(nrs, input_shape=(columns,), activation='relu', name='input', activity_regularizer=l1(0.001)),
                tf.keras.layers.Dropout(0.2, name='do1'),
                tf.keras.layers.Dense(nrs, activation='relu', name='fc1'),
                tf.keras.layers.Dropout(0.2, name='do2'),
                tf.keras.layers.Dense(3, activation='softmax', name='output'),
            ])
            optimizer = tf.keras.optimizers.Adam(lr=lr)
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

def get_best_hyperparametersKFold(train_x, train_y, k):
    # eliminamos los datos de la carpeta de logs de callbacks
    # shutil.rmtree('trainingLogs/')
    # definimos los parámetros a entrenar con el modelo
    learn_rate = LEARN_RATES
    # el número de neuronas por capa debe de ser bajo para no producir overfitting
    # si el número de neuronas es alto la diferencia entre el acc del train y del dev son muy grandes
    # por lo que el acc del modelo no será el real
    # tener más neuronas por capa hará el modelo más complejo y a su vez más inestable(muchas subidas y bajadas del acc)
    # neurons = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    # entrenameros con neuronas en el rango 4-64 para obtener modelos más estables
    neurons = NEURONS_PER_LAYER
    best_learn_rate = learn_rate[0]
    best_neurons_per_layer = neurons[0]
    best_accuracy = 0
    best_model = None
    # fix random seed for reproducibility
    seed = 7
    columns = np.size(train_x, 1)
    print(columns)
    for lr in learn_rate:
        for nrs in neurons:
            # definimos K-fold cross validation
            kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            accscores = []
            lossscores = []
            i = 1
            for train, dev in kfold.split(train_x, train_y):
                directory_name = str(lr), "_", str(nrs),"_",str(i)
                i = i + 1
                # haremos un early stopping si en las siguientes 3 epocs del modelo no se produce una mejora
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
                training_tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir="trainingLogs\{}".format(directory_name))
                # añadiremos una capa dropout para evitar overfitting
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(nrs, input_shape=(columns,), activation='relu', name='input',
                                          activity_regularizer=l1(0.001)),
                    tf.keras.layers.Dropout(0.2, name='do1'),
                    tf.keras.layers.Dense(nrs, activation='relu', name='fc1'),
                    tf.keras.layers.Dropout(0.2, name='do2'),
                    tf.keras.layers.Dense(3, activation='softmax', name='output'),
                ])
                optimizer = tf.keras.optimizers.Adam(lr=lr)
                model.compile(loss=LOSS, optimizer=optimizer, metrics=['accuracy'])
                model.fit(train_x[train], train_y[train], epochs=EPOCHS, verbose=0, callbacks=[training_tensorboard_callback, early_stopping_callback], validation_data=(train_x[dev], train_y[dev]))
                # Para poder observar el entrenamiento de los datos en TB: tensorboard --logdir=C:\Users\frank\PycharmProjects\TFGEnglishMining\src\trainingLogs\
                # Evaluamos el modelo obtenido
                scores = model.evaluate(train_x[dev], train_y[dev], verbose=0)
                # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                lossscores.append(scores[0])
                accscores.append(scores[1])
            meanAccuracy = np.mean(accscores)
            meanLoss = np.mean(lossscores)
            print('Dev set loss: {0}, Dev set accuracy: {1}, Learning rate: {2}, Neurons: {3}'.format(meanLoss, meanAccuracy, lr, nrs))
            # comprobamos si hemos obtenido un mejor accuracy
            if best_accuracy < meanAccuracy:
                best_accuracy = meanAccuracy
                best_learn_rate = lr
                best_neurons_per_layer = nrs
                best_model = model

    print('Final best Acc: {0}, Final best learning_rate {1}, Final best neurons per layer {2}'.format(best_accuracy, best_learn_rate, best_neurons_per_layer))

    return best_learn_rate, best_neurons_per_layer, best_model

def save_model(model, targetPath):
    # guardamos el modelo
    # model.save('mi_modelo.h5')  # creamos un archivo HDF5 'my_model.h5'
    tf.keras.models.save_model(model, targetPath, save_format='h5')  # creamos un archivo HDF5 'my_model.h5'

if __name__ == "__main__":
    main()
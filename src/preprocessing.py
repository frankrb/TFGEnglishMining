import pandas as pd
import csv as csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing

def main():
    # cambiamos las clases de Train y Test a números enteros
    change_class_to_int("../data/Test_Aztertest/test_aztertest.csv", "../data/DataExamples/test_class_integer.csv")
    change_class_to_int("../data/Train_Aztertest/train_aztertest.csv", "../data/DataExamples/train_class_integer.csv")
    # ordenamos las columnas de train y test para que tengan el mismo orden
    order_columns("../data/DataExamples/train_class_integer.csv", "../data/DataExamples/test_class_integer.csv", "../data/DataExamples/test.csv")
    # creamos un train y un dev(X% del train) equilibrado partiendo del train anterior
    create_train_dev(20, "../data/DataExamples/train_class_integer.csv", "../data/DataExamples/train.csv", "../data/DataExamples/dev.csv")

    # fchiSquared(100, "../data/Train/train.csv", "../data/Dev/dev.csv","../data/Test/test.csv")
    # fClassif(100, "../data/Train/train.csv", "../data/Dev/dev.csv", "../data/Test/test.csv")
    # fmutual_info_classif(100, "../data/Train/train.csv", "../data/Dev/dev.csv", "../data/Test/test.csv")

    # cargamos los datos para poder trabajar con ellos
    train_x, train_y, dev_x, dev_y, test_x, test_y = fLoadTrainDevTestNormalizing("../data/DataExamples/train.csv", "../data/DataExamples/dev.csv","../data/DataExamples/test.csv")

    # hacemos una selección de atributos (100 mejores): chi2, f_classif, mutual_info_classif
    train_x, dev_x, test_x = fSelectKbestNormalize(100, "chi2", train_x, train_y, dev_x, test_x)
    print("FIN")

def order_columns(pTrainPath, pTestPathSource, pTestPathTarget):
    # train y test deben de tener las mismas columnas
    # reordenar las columnas del TEST como el TRAIN y lo guardamos
    train_example = pd.read_csv(pTrainPath)
    columns = train_example.columns.values.tolist()
    dataframe = pd.read_csv(pTestPathSource)
    dataframe = dataframe[columns]
    # print(columns)
    dataframe.to_csv(pTestPathTarget, header=True, index=None)

def fLoadTrainDevTestNormalizing(pTrainPath, pDevPath, pTestPath):
    # load train dataset
    dataframe = pd.read_csv(pTrainPath, header=None, skiprows=1)
    dataset = dataframe.values
    train_x = dataset[:, 0:148].astype(float)
    train_y = dataframe.iloc[:, -1:].values

    # load dev dataset
    dataframe = pd.read_csv(pDevPath, header=None, skiprows=1)
    dataset = dataframe.values
    dev_x = dataset[:, 0:148].astype(float)
    dev_y = dataframe.iloc[:, -1:].values

    # load test dataset
    dataframe = pd.read_csv(pTestPath, header=None, skiprows=1)
    dataset = dataframe.values
    test_x = dataset[:, 0:148].astype(float)
    test_y = dataframe.iloc[:, -1:].values

    # normalizamos los datos de las features (los valores de las clases se mantienen igual)
    train_x = preprocessing.normalize(train_x)
    dev_x = preprocessing.normalize(dev_x)
    test_x = preprocessing.normalize(test_x)

    return train_x, train_y, dev_x, dev_y, test_x, test_y

def fLoadTrainTestNormalizing(pTrainPath, pTestPath):
    # load train dataset
    dataframe = pd.read_csv(pTrainPath, header=None, skiprows=1)
    dataset = dataframe.values
    train_x = dataset[:, 0:148].astype(float)
    train_y = dataframe.iloc[:, -1:].values

    # load test dataset
    dataframe = pd.read_csv(pTestPath, header=None, skiprows=1)
    dataset = dataframe.values
    test_x = dataset[:, 0:148].astype(float)
    test_y = dataframe.iloc[:, -1:].values

    # normalizamos los datos de las features (los valores de las clases se mantienen igual)
    train_x = preprocessing.normalize(train_x)
    test_x = preprocessing.normalize(test_x)

    return train_x, train_y, test_x, test_y

def fLoadNormalizing(pSourcePath):
    # cargamos los datos normalizando
    # load dataset
    dataframe = pd.read_csv(pSourcePath, header=None, skiprows=1)
    dataset = dataframe.values
    data_x = dataset[:, 0:148].astype(float)
    data_y = dataframe.iloc[:, -1:].values
    # normalizamos los datos de las features (los valores de las clases se mantienen igual)
    data_x = preprocessing.normalize(data_x)
    return data_x, data_y

def fSelectKbestNormalize(pNumFeatures, pType, pTrainX, pTrainY, pDevX, pTestX):
    if pType == 'chi2':
        selector = SelectKBest(chi2, k=pNumFeatures)
    elif pType == 'f_classif':
        selector = SelectKBest(f_classif, k=pNumFeatures)
    elif pType == 'mutual_info_classif':
        selector = SelectKBest(mutual_info_classif, k=pNumFeatures)
    else:
        exit('Error1: Método de selección de atributos incorrecto')
    pTrainX = selector.fit_transform(pTrainX, pTrainY)
    pDevX = selector.transform(pDevX)
    pTestX = selector.transform(pTestX)

    return pTrainX, pDevX, pTestX

def fSelectKbestNormalize1(pNumFeatures, pType, pTrainX, pTrainY, pTestX):
    if pType == 'chi2':
        selector = SelectKBest(chi2, k=pNumFeatures)
    elif pType == 'f_classif':
        selector = SelectKBest(f_classif, k=pNumFeatures)
    elif pType == 'mutual_info_classif':
        selector = SelectKBest(mutual_info_classif, k=pNumFeatures)
    else:
        exit('Error: Método de selección de atributos incorrecto')
    pTrainX = selector.fit_transform(pTrainX, pTrainY)
    pTestX = selector.transform(pTestX)

    return pTrainX, pTestX

def change_class_to_int(csv_source_path,csv_target_path):
    r = csv.reader(open(csv_source_path))
    lines = list(r)
    num_rows = len(lines)
    num_cols = len(lines[0])
    class_index = num_cols - 1
    #print(num_rows)
    #print(num_cols)
    #print(class_index)
    i = 0
    while i < num_rows-1:
        i = i + 1
        #print(i)
        #print(lines[i][class_index])
        if lines[i][class_index] == 'elementary':
            lines[i][class_index] = 0
        if lines[i][class_index] == 'intermediate':
            lines[i][class_index] = 1
        if lines[i][class_index] == 'advanced':
            lines[i][class_index] = 2


    writer = csv.writer(open(csv_target_path, 'w', newline=''))
    writer.writerows(lines)

def create_train_dev(pDevPercent, pTrainSourcePath, pTrainTargetPath, pDevTargetPath):
    # esta función nos permite crear un train y un dev equilibrado
    # de este modo se obtiene un train-dev-test con equilibrio
    # las clases deben de estar dadas por números (0, 1, 2)
    train = 0
    test = 0
    data = pd.read_csv(pTrainSourcePath)
    # print(data.groupby('level').count())
    num_rows_0 = data.groupby('level').count().iloc[0, -1]
    num_rows_1 = data.groupby('level').count().iloc[1, -1]
    num_rows_2 = data.groupby('level').count().iloc[2, -1]

    num_rows_0 = num_rows_0*pDevPercent//100
    num_rows_1 = num_rows_1*pDevPercent//100
    num_rows_2 = num_rows_2*pDevPercent//100
    # creamos un csv train y dev
    data_train = pd.read_csv(pTrainSourcePath)
    data_test = pd.read_csv(pTrainSourcePath)
    i = 0
    data_train_drop = []
    data_test_drop = []
    total_rows = len(data.index)
    # print(total_rows)
    # print(data.iloc[455, -1])
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
    # print(data_train_drop)
    # print(data_test_drop)
    data_train.drop(data_train_drop, axis=0, inplace=True)
    data_test.drop(data_test_drop, axis=0, inplace=True)

    data_test.to_csv(pDevTargetPath, header=True, index=None)
    data_train.to_csv(pTrainTargetPath, header=True, index=None)

def floadData(data_source_path, data_target_path):
    # cargamos los datos en el path target
    data = pd.read_csv(data_source_path)
    data.to_csv(data_target_path, header=True, index=None)
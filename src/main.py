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
import tensorflow_datasets as tfds
import csv

def main():
    print("Ejecutando Main...")

    """ 
    # código para cargar train y test desde una URL
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
    """

    # ponemos el path de los datos de train y test

    train_file_path = "../data/Temporales/train_clean.csv"
    test_file_path = "../data/Temporales/test_clean.csv"

    # change_class_to_int()

    #check_CSV(train_file_path)

    # seleccionamos las comulnas(atributos) que vamos a usar.
    CSV_COLUMNS = ['num_words','num_paragraphs','num_sentences','num_words_with_punct','num_total_prop','num_noun','num_verb','num_past','num_rare_words_4','num_rare_nouns_4','num_dif_rare_words_4','num_pres','num_indic','num_adv','num_personal_pronouns','num_third_pers_pron','num_subord','num_inf','num_adj','num_rel_subord','num_ger','num_past_irregular','num_rare_verbs_4','num_impera','num_rare_adj_4','num_pass','num_agentless','num_first_pers_pron','num_neg','noun_overlap_adjacent','noun_overlap_all','argument_overlap_adjacent','argument_overlap_all','stem_overlap_adjacent','stem_overlap_all','content_overlap_adjacent_mean','content_overlap_adjacent_std','content_overlap_all_mean','content_overlap_all_std','all_connectives_incidence','causal_connectives_incidence','logical_connectives_incidence','adversative_connectives_incidence','temporal_connectives_incidence','conditional_connectives_incidence','dale_chall','smog','flesch_kincaid','flesch','simple_ttr','nttr','vttr','adj_ttr','adv_ttr','content_ttr','lemma_ttr','lemma_nttr','lemma_vttr','lemma_adj_ttr','lemma_adv_ttr','lemma_content_ttr','honore','maas','mtld','sentences_per_paragraph_mean','sentences_length_mean','sentences_length_no_stopwords_mean','num_syllables_words_mean','words_length_mean','words_length_no_stopwords_mean','lemmas_length_mean','sentences_per_paragraph_std','sentences_length_std','sentences_length_no_stopwords_std','num_syllables_words_std','words_length_std','words_length_no_stopwords_std','lemmas_length_std','left_embeddedness','num_content_words_not_a1_c1_words','num_a1_words','num_b2_words','num_b1_words','num_a2_words','num_c1_words','mean_depth_per_sentence','num_different_forms','num_pass_mean','num_past_irregular_mean','num_punct_marks_per_sentence','mean_propositions_per_sentence','mean_vp_per_sentence','mean_np_per_sentence','noun_phrase_density_incidence','verb_phrase_density_incidence','lexical_density','noun_density','verb_density','adj_density','adv_density','agentless_passive_density_incidence','negation_density_incidence','gerund_density_incidence','infinitive_density_incidence','num_modifiers_noun_phrase','num_decendents_noun_phrase','mean_rare_4','mean_distinct_rare_4','min_wf_per_sentence','polysemic_index','hypernymy_index','hypernymy_verbs_index','hypernymy_nouns_index','num_paragraphs_incidence','num_sentences_incidence','num_past_incidence','num_pres_incidence','num_future','num_future_incidence','num_indic_incidence','num_impera_incidence','num_past_irregular_incidence','num_pass_incidence','num_rare_nouns_4_incidence','num_rare_adj_4_incidence','num_rare_verbs_4_incidence','num_rare_advb_4','num_rare_advb_4_incidence','num_rare_words_4_incidence','num_dif_rare_words_4_incidence','num_lexic_words_incidence','num_noun_incidence','num_adj_incidence','num_adv_incidence','num_verb_incidence','num_subord_incidence','num_rel_subord_incidence','num_personal_pronouns_incidence','num_first_pers_pron_incidence','num_first_pers_sing_pron','num_first_pers_sing_pron_incidence','num_third_pers_pron_incidence','num_a1_words_incidence','num_a2_words_incidence','num_b1_words_incidence','num_b2_words_incidence','num_c1_words_incidence','num_content_words_not_a1_c1_words_incidence']

    # cargamos los datos de train para entrenar nuestro modelo
    raw_train_data = get_dataset(train_file_path)
    raw_test_data = get_dataset(test_file_path)

    # Mostramos los datos

    show_batch(raw_train_data)

    preprocessing_layer = tf.keras.layers.DenseFeatures(numeric_columns)

    exit(1)

    examples, labels = next(iter(raw_train_data))  # Just the first batch.
    print("EXAMPLES: \n", examples, "\n")
    print("LABELS: \n", labels)

    #obetenemos las medias de cada columna y las guardamos en MEANS con el nombre de cada columna como clave
    means = get_means(pd.read_csv(train_file_path),CSV_COLUMNS)

    print(means)

    MEANS = {}
    keys = CSV_COLUMNS
    values = means
    j = 0
    for i in keys:
        MEANS[i] = values[j]
        j=j+1
    print(MEANS)

    numerical_columns = []
    for feature in MEANS.keys():
        num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data,
                                                                                            MEANS[feature]))
        numerical_columns.append(num_col)

    print(numerical_columns)

    #creamos un modelo

    preprocessing_layer = tf.keras.layers.DenseFeatures(numerical_columns)

    model = get_compiled_model(preprocessing_layer)

    #entrenamos || evaluamos || predecimos

    raw_train_data.skip(1)
    train_data = raw_train_data.shuffle(500)
    test_data = raw_test_data

    model.fit(train_data, epochs=30)

    test_loss, test_accuracy = model.evaluate(test_data)

    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

    # Show some results
    outcome = "NONE"
    predictions = model.predict(test_data)
    print(predictions)

    for prediction, level in zip(predictions[:10], list(test_data)[0][1][:10]):
        print("Predicted level: {:.2%}".format(prediction[0]), " | Actual outcome: ", level)


def get_dataset(file_path):
    # leer csf usando CSV Tensorflow
    # es necesario identificar la columna de la clase y sus valores
    LABEL_COLUMN = 'level'
    LABELS = [0, 1, 2]


    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=137,  # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        header=True,
        na_value="?",
        num_epochs=1,
        ignore_errors=True)


    # dataset.filter()
    return dataset

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))


def check_CSV(csv_dataset):
    # comprobamos cada línea del CSV
    print("probando datos. INICIO")
    reader = csv.DictReader(open(csv_dataset))
    i = 0

    for raw in reader:
        # print(len(raw) + ' ' + raw)
        i = i + 1
        print("{} {} ()".format(i, len(raw), raw.values()))

    print("probando datos. FIN")

def process_continuous_data(mean, data):
  # Normalize data

  data = tf.cast(data, tf.float32) * 1/(2*mean)

  return tf.reshape(data, [-1, 1])

def get_compiled_model(preprocessing_layer):

    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(1, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model

def get_means(csv,CSV_COLUMNS):

    i=0;
    means=[]
    for columnas in CSV_COLUMNS:
        means.append(csv[columnas].mean())
        print(csv[columnas].mean())
        i=i+1

    return  means


def change_class_to_int():
    CSV_COLUMNS = ['num_words', 'num_paragraphs', 'num_sentences', 'num_words_with_punct', 'num_total_prop', 'num_noun',
                   'num_verb', 'num_past', 'num_rare_words_4', 'num_rare_nouns_4', 'num_dif_rare_words_4', 'num_pres',
                   'num_indic', 'num_adv', 'num_personal_pronouns', 'num_third_pers_pron', 'num_subord', 'num_inf',
                   'num_adj', 'num_rel_subord', 'num_ger', 'num_past_irregular', 'num_rare_verbs_4', 'num_impera',
                   'num_rare_adj_4', 'num_pass', 'num_agentless', 'num_first_pers_pron', 'num_neg',
                   'noun_overlap_adjacent', 'noun_overlap_all', 'argument_overlap_adjacent', 'argument_overlap_all',
                   'stem_overlap_adjacent', 'stem_overlap_all', 'content_overlap_adjacent_mean',
                   'content_overlap_adjacent_std', 'content_overlap_all_mean', 'content_overlap_all_std',
                   'all_connectives_incidence', 'causal_connectives_incidence', 'logical_connectives_incidence',
                   'adversative_connectives_incidence', 'temporal_connectives_incidence',
                   'conditional_connectives_incidence', 'dale_chall', 'smog', 'flesch_kincaid', 'flesch', 'simple_ttr',
                   'nttr', 'vttr', 'adj_ttr', 'adv_ttr', 'content_ttr', 'lemma_ttr', 'lemma_nttr', 'lemma_vttr',
                   'lemma_adj_ttr', 'lemma_adv_ttr', 'lemma_content_ttr', 'honore', 'maas', 'mtld',
                   'sentences_per_paragraph_mean', 'sentences_length_mean', 'sentences_length_no_stopwords_mean',
                   'num_syllables_words_mean', 'words_length_mean', 'words_length_no_stopwords_mean',
                   'lemmas_length_mean', 'sentences_per_paragraph_std', 'sentences_length_std',
                   'sentences_length_no_stopwords_std', 'num_syllables_words_std', 'words_length_std',
                   'words_length_no_stopwords_std', 'lemmas_length_std', 'left_embeddedness',
                   'num_content_words_not_a1_c1_words', 'num_a1_words', 'num_b2_words', 'num_b1_words', 'num_a2_words',
                   'num_c1_words', 'mean_depth_per_sentence', 'num_different_forms', 'num_pass_mean',
                   'num_past_irregular_mean', 'num_punct_marks_per_sentence', 'mean_propositions_per_sentence',
                   'mean_vp_per_sentence', 'mean_np_per_sentence', 'noun_phrase_density_incidence',
                   'verb_phrase_density_incidence', 'lexical_density', 'noun_density', 'verb_density', 'adj_density',
                   'adv_density', 'agentless_passive_density_incidence', 'negation_density_incidence',
                   'gerund_density_incidence', 'infinitive_density_incidence', 'num_modifiers_noun_phrase',
                   'num_decendents_noun_phrase', 'mean_rare_4', 'mean_distinct_rare_4', 'min_wf_per_sentence',
                   'polysemic_index', 'hypernymy_index', 'hypernymy_verbs_index', 'hypernymy_nouns_index',
                   'num_paragraphs_incidence', 'num_sentences_incidence', 'num_past_incidence', 'num_pres_incidence',
                   'num_future', 'num_future_incidence', 'num_indic_incidence', 'num_impera_incidence',
                   'num_past_irregular_incidence', 'num_pass_incidence', 'num_rare_nouns_4_incidence',
                   'num_rare_adj_4_incidence', 'num_rare_verbs_4_incidence', 'num_rare_advb_4',
                   'num_rare_advb_4_incidence', 'num_rare_words_4_incidence', 'num_dif_rare_words_4_incidence',
                   'num_lexic_words_incidence', 'num_noun_incidence', 'num_adj_incidence', 'num_adv_incidence',
                   'num_verb_incidence', 'num_subord_incidence', 'num_rel_subord_incidence',
                   'num_personal_pronouns_incidence', 'num_first_pers_pron_incidence', 'num_first_pers_sing_pron',
                   'num_first_pers_sing_pron_incidence', 'num_third_pers_pron_incidence', 'num_a1_words_incidence',
                   'num_a2_words_incidence', 'num_b1_words_incidence', 'num_b2_words_incidence',
                   'num_c1_words_incidence', 'num_content_words_not_a1_c1_words_incidence','level']


    class_index = len(CSV_COLUMNS) - 1
    print(class_index)

    r = csv.reader(open('../data/Test_Aztertest/test_aztertest.csv'))  # Here your csv file
    lines = list(r)
    num_rows = len(lines)
    num_cols = len(lines[0])
    print(num_rows)
    print(num_cols)
    i = 0
    while i < num_rows-1:
        i = i + 1
        print(i)
        print(lines[i][class_index])
        if lines[i][class_index] == 'advanced':
            lines[i][class_index] = 0
        if lines[i][class_index] == 'intermediate':
            lines[i][class_index] = 1
        if lines[i][class_index] == 'elementary':
            lines[i][class_index] = 2


    writer = csv.writer(open('../data/Temporales/test_clean.csv', 'w', newline=''))
    writer.writerows(lines)



if __name__ == '__main__':
    main()



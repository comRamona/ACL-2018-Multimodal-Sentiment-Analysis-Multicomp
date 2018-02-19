#!/usr/bin/env python

'''
This script shows you how to:
1. Load the features for each segment;
2. Load the labels for each segment;
3. Perform word level feature alignment;
3. Use Keras to implement a simple LSTM on top of the data
'''
###
from __future__ import print_function
import numpy as np
import pandas as pd
import random as rn
import os, sys
import tensorflow as tf
import argparse
from collections import defaultdict
from utils.parser_utils import KerasParserClass
from utils.storage import build_experiment_folder, save_statistics

parser = argparse.ArgumentParser(description='Welcome to TFN experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, batch_norm, \
experiment_prefix, dropout_rate, n_layers, max_len = parser_extractor.get_argument_variables()


experiment_name = "experiment_{}_batch_size_{}_bn_{}_dr{}_nl_{}_ml_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   dropout_rate, n_layers, max_len)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = '0'
tf.set_random_seed(seed)

###

max_len = 15

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, Merge, Input, Reshape
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras.regularizers import l1, l2
from keras.constraints import nonneg
from sklearn.metrics import mean_absolute_error

from mmdata import Dataloader, Dataset, MOSI
from sklearn.svm import SVR,SVC
from sklearn.metrics import accuracy_score,  confusion_matrix, precision_recall_fscore_support, mean_squared_error, classification_report, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
import scipy
import scipy.stats 
from videodata import VideoData
target_names = ['strg_neg', 'weak_neg', 'neutral', 'weak_pos', 'strg_pos']

video = VideoData()
x_train, x_valid, x_test, y_train, y_valid, y_test = video.get_video(max_len)

print("Data preprocessing finished! Begin compiling and training model.")
print(x_train.shape)
print(x_valid.shape)

# sort through all the video ID, segment ID pairs
facet = video.dataset.facet()
train_set_ids = []
for vid in x_train:
    for sid in facet['facet'][vid].keys():
        train_set_ids.append((vid, sid))

valid_set_ids = []
for vid in valid_ids:
    for sid in facet['facet'][vid].keys():
        valid_set_ids.append((vid, sid))

test_set_ids = []
for vid in test_ids:
    for sid in facet['facet'][vid].keys():
        test_set_ids.append((vid, sid))

max_len = 15

# binarize the sentiment scores for binary classification task
y_train_bin = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
y_valid_bin = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0
y_test_bin = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

# for regression
y_train_reg = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
y_valid_reg = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
y_test_reg = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])


x_train[x_train != x_train] = 0
x_valid[x_valid != x_valid] = 0
x_test[x_test != x_test] = 0

end_to_end = True
f_Covarep_num = x_train.shape[1]
Covarep_model = Sequential()
Covarep_model.add(BatchNormalization(input_shape=(f_Covarep_num,), name = 'covarep_layer_0'))
Covarep_model.add(Dropout(0.2, name = 'covarep_layer_1'))
Covarep_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'covarep_layer_2', trainable=end_to_end))
Covarep_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'covarep_layer_3', trainable=end_to_end))
Covarep_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'covarep_layer_4', trainable=end_to_end))
Covarep_model.add(Dense(1, name = 'covarep_layer_5'))

train_patience = 5
callbacks = [
        EarlyStopping(monitor='val_loss', patience=train_patience, verbose=0),
    ]
momentum = 0.9
lr = 0.01
train_epoch = 5
loss = "mae"
opt = "adam"
sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
adam = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #decay=0.999)
optimizer = {'sgd': sgd, 'adam':adam}
Covarep_model.compile(loss=loss, optimizer=optimizer[opt])
Covarep_model.fit(x_train, y_train_reg, validation_data=(x_valid,y_valid_reg), nb_epoch=train_epoch, batch_size=128, callbacks=callbacks)
predictions = Covarep_model.predict(x_test, verbose=0)
predictions = predictions.reshape((len(y_test_reg),))
y_test = y_test_reg.reshape((len(y_test_reg),))
mae = np.mean(np.absolute(predictions-y_test))
print("mae: "+str(mae))
print("corr: "+str(round(np.corrcoef(predictions,y_test)[0][1],5)))
print("mult_acc: "+str(round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)))
true_label = (y_test >= 0)
predicted_label = (predictions >= 0)
print("Confusion Matrix :")
print(confusion_matrix(true_label, predicted_label))
print("Classification Report :")
print(classification_report(true_label, predicted_label, digits=5))
print("Accuracy ", accuracy_score(true_label, predicted_label))


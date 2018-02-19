#!/usr/bin/env python

'''
This script shows you how to:
1. Load the features for each segment;
2. Load the labels for each segment;
3. Prerocess data and use Keras to implement a simple LSTM on top of the data
'''

from __future__ import print_function
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from mmdata import MOSEI
from utils.parser_utils import KerasParserClass
from utils.storage import build_experiment_folder, save_statistics

parser = argparse.ArgumentParser(description='Welcome to LSTM experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, batch_norm, \
experiment_prefix, dropout_rate, n_layers, max_len = parser_extractor.get_argument_variables()


experiment_name = "experiment_{}_batch_size_{}_bn_{}_dr{}_nl_{}_ml_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   dropout_rate, n_layers, max_len)
#  generate experiment name
np.random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
tf.set_random_seed(seed)
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.


# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

def pad(data, max_len):
    """A funtion for padding/truncating sequence data to a given lenght"""
    # recall that data at each time step is a tuple (start_time, end_time, feature_vector), we only take the vector
    data = np.array([feature[2] for feature in data])
    n_rows = data.shape[0]
    dim = data.shape[1]
    if max_len >= n_rows:
        diff = max_len - n_rows
        padding = np.zeros((diff, dim))
        padded = np.concatenate((padding, data))
        return padded
    else:
        return data[-max_len:]

if __name__ == "__main__":
    # Download the data if not present
    mosi = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSI')
    facet = mosi.facet()
    sentiments = mosi.sentiments() # sentiment labels, real-valued. for this tutorial we'll binarize them
    train_ids = mosi.train() # set of video ids in the training set
    test_ids = mosi.valid() # set of video ids in the valid set
#    test_ids = mosi.test() # set of video ids in the test set


    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
        for sid in facet['facet'][vid].keys():
            train_set_ids.append((vid, sid))

    test_set_ids = []
    for vid in test_ids:
        for sid in facet['facet'][vid].keys():
            test_set_ids.append((vid, sid))


    # partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
    # data will have shape (dataset_size, max_len, feature_dim)
    max_len = 20

    train_set_video = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids if facet['facet'][vid][sid]], axis=0)
#    valid_set_video = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['facet'][vid][sid]], axis=0)
    test_set_video = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids if facet['facet'][vid][sid]], axis=0)

    # binarize the sentiment scores for binary classification task
    y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
    y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

    # normalize facet features, remove possible NaN values
    video_max = np.max(np.max(np.abs(train_set_video), axis=0), axis=0)
    train_set_video = train_set_video / video_max
    valid_set_video = valid_set_video / video_max
    test_set_video = test_set_video / video_max

    train_set_video[train_set_video != train_set_video] = 0
    valid_set_video[valid_set_video != valid_set_video] = 0
    test_set_video[test_set_video != test_set_video] = 0

    model = Sequential()
    model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))


    # you can try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['binary_accuracy'])
    batch_size = 32

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=20,
            validation_data=[x_valid, y_valid])

    # evaluate
    y_preds = model.predict(x_test)
    test_acc = np.mean((y_preds > 0.5) == y_test.reshape(-1, 1))
    print("The accuracy on test set is: {}".format(test_acc))


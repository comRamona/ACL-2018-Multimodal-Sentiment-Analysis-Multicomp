#!/usr/bin/env python

'''
This script shows you how to:
1. Load the features for each segment;
2. Load the labels for each segment;
3. Perform word level feature alignment;
3. Use Keras to implement a simple LSTM on top of the data
'''

import numpy as np
import pandas as pd
import pdb
from collections import defaultdict
from mmdata import MOSI
import argparse
from collections import defaultdict
from mmdata.dataset import Dataset
from utils.parser_utils import KerasParserClass
from utils.storage import build_experiment_folder, save_statistics

parser = argparse.ArgumentParser(description='Welcome to LSTM experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, batch_norm, \
experiment_prefix, dropout_rate, n_layers, max_len = parser_extractor.get_argument_variables()


experiment_name = "experiment_{}_batch_size_{}_bn_{}_dr{}_nl_{}_ml_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   dropout_rate, n_layers, max_len)
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
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization
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
    max_len = 20
    mosi = MOSI()
    embeddings = mosi.embeddings()
    facet = mosi.facet()
    covarep = mosi.covarep()
    sentiments = mosi.sentiments() # sentiment labels, real-valued. for this tutorial we'll binarize them
    train_ids = mosi.train()
    valid_ids = mosi.valid()
    test_ids = mosi.test()

    # Merge different features and do word level feature alignment (align according to timestamps of embeddings)
    bimodal = Dataset.merge(embeddings, facet)
    trimodal = Dataset.merge(bimodal, covarep)
    dataset = trimodal.align('embeddings')

    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
        for sid in dataset['embeddings'][vid].keys():
            if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
                train_set_ids.append((vid, sid))

    valid_set_ids = []
    for vid in valid_ids:
        for sid in dataset['embeddings'][vid].keys():
            if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
                valid_set_ids.append((vid, sid))

    test_set_ids = []
    for vid in test_ids:
        for sid in dataset['embeddings'][vid].keys():
            if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
                test_set_ids.append((vid, sid))


    # partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
    # data will have shape (dataset_size, max_len, feature_dim)

    train_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['covarep'][vid][sid]], axis=0)
    valid_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['covarep'][vid][sid]], axis=0)
    
    train_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
    valid_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)

    train_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
    valid_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)

    # binarize the sentiment scores for binary classification task
    y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
    y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0
    y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

    train_set_audio = train_set_audio[:,:,1:35]
    valid_set_audio = valid_set_audio[:,:,1:35]
    train_set_visual = train_set_visual[:, :, 1:35]
    valid_set_visual = valid_set_visual[:, :, 1:35]

    visual_max = np.max(np.max(np.abs(train_set_visual), axis=0), axis=0)
    visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize this dimension
    train_set_visual = train_set_visual / visual_max
    valid_set_visual = valid_set_visual / visual_max


    train_set_visual[train_set_visual != train_set_visual] = 0
    valid_set_visual[valid_set_visual != valid_set_visual] = 0


    audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
    audio_max[audio_max==0] = 1
    train_set_audio = train_set_audio / audio_max
    valid_set_audio = valid_set_audio / audio_max


    train_set_audio[train_set_audio != train_set_audio] = 0
    valid_set_audio[valid_set_audio != valid_set_audio] = 0
    
    #pdb.set_trace()

    # early fusion: input level concatenation of features
    x_train = np.concatenate((train_set_visual, train_set_audio, train_set_text), axis=2)
    x_valid = np.concatenate((valid_set_visual, valid_set_audio, valid_set_text), axis=2)

    model = Sequential()
    model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))


    # you can try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    batch_size = 32

    saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)  # generate experiment dir
    filepath = "{}/best_validation_{}".format(saved_models_filepath, experiment_name) + ".ckpt"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_acc',
                                  min_delta=0,
                                  patience=10,
                                  verbose=1, mode='auto')
    tensor_board = TensorBoard(log_dir=logs_filepath, histogram_freq=0, batch_size=batch_size, write_graph=True, 
        write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [checkpoint, early_stopping, tensor_board]
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_valid, y_valid],
              callbacks=callbacks_list)
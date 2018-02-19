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
from mmdata import MOSI
from videodata import VideoData
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

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

print("Preparing train and test data...")
# Download the data if not present
mosi = MOSI()
facet = mosi.facet()
sentiments = mosi.sentiments()
train_ids = mosi.train()
valid_ids = mosi.valid()
test_ids = mosi.test()
max_len = 20
x_train = []
y_train = []
x_val = []
y_val = []

video = VideoData()
x_train, x_valid, x_test, y_train, y_valid, y_test = video.get_video(max_len)

print("Data preprocessing finished! Begin compiling and training model.")
print(x_train.shape)
print(x_valid.shape)


    # partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
    # data will have shape (dataset_size, max_len, feature_dim)

    #train_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
    #valid_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
    #test_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)
# Prepare the final inputs as numpy arrays
x_train = np.asarray(x_train)
x_val = np.asarray(x_val)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
print("Data preprocessing finished! Begin compiling and training model.")

print(max_len)
model = Sequential()

k = 3
m = 2
model.add(Conv1D(filters=128, kernel_size=3, input_shape = (max_len, 300), activation='relu'))
model.add(MaxPooling1D(2))

#model.add(MaxPooling1D(m))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)  # generate experiment dir


# try using different optimizers and different optimizer configs
optimizer = Adam(lr=0.0001)
model.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])
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
          validation_data=[x_val, y_val],
          callbacks=callbacks_list)


# mosei = MOSI()
# embeddings = mosei.embeddings()
# sentiments = mosei.sentiments()
# train_ids = mosei.train()
# valid_ids = mosei.valid()
# embeddings = mosei.embeddings()
# train_set_ids = []
# for vid in train_ids:
#     for sid in embeddings['embeddings'][vid].keys():
#         if embeddings['embeddings'][vid][sid]:
#             train_set_ids.append((vid, sid))

# train_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)

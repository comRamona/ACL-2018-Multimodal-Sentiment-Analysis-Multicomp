#!/usr/bin/env python

'''
This script shows you how to:
1. Load the features for each segment;
2. Load the labels for each segment;
3. Prerocess data and use Keras to implement a simple LSTM on top of the data
'''

###
from __future__ import print_function
import numpy as np
import pandas as pd
import random as rn
import os
import tensorflow as tf
import argparse
from collections import defaultdict
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
os.environ['PYTHONHASHSEED'] = '0'
tf.set_random_seed(seed)

###

max_len = 20

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from mmdata import MOSI, Dataloader, Dataset
from videodata import VideoData

video = VideoData()
x_train, x_valid, x_test, y_train, y_valid, y_test = video.get_video(max_len)

print("Data preprocessing finished! Begin compiling and training model.")
print(x_train.shape)
print(x_valid.shape)

model = Sequential()
model.add(Bidirectional(LSTM(64), input_shape=(max_length,300))
#model.add(Dropout(0.5))
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


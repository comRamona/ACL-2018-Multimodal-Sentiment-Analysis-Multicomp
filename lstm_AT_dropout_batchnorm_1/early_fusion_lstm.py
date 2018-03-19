#!/usr/bin/env python

'''
This script shows you how to:
1. Load the features for each segment;
2. Load the labels for each segment;
3. Perform word level feature alignment;
3. Use Keras to implement a simple LSTM on top of the data
'''

import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from mmdata import MOSI
import argparse
from collections import defaultdict
from mmdata.dataset import Dataset
from utils.parser_utils import KerasParserClass
from utils.storage import build_experiment_folder, save_statistics

parser = argparse.ArgumentParser(description='CNN experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input


#experiment_prefix = sys.argv[1] #cnn_early_fusion
#max_len = int(sys.argv[2]) #[15, 20, 25]
#dropout_rate = float(sys.argv[4]) # [0.1, 0.2, 0.35]
#n_layers = int(sys.argv[5]) # [1, 2, 3]
#epochs = int(sys.argv[6]) # [50, 100]

batch_size, seed, epochs, logs_path, mode,continue_from_epoch, batch_norm, \
experiment_prefix, dropout_rate, n_layers, max_len = parser_extractor.get_argument_variables()


experiment_name = "{}_m_{}_ep_{}_bs_{}_bn_{}_dr_{}_nl_{}_ml_{}".format(experiment_prefix, mode,epochs,
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
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Conv2D, Flatten,BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger

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
        idx = np.random.choice(np.arange(n_rows), max_len, replace=False)
        return data[idx]

if __name__ == "__main__":
    # Download the data if not present
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
    test_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['covarep'][vid][sid]], axis=0)

    train_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
    valid_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
    test_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

    train_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
    valid_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
    test_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

    # binarize the sentiment scores for binary classification task
    y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
    y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0
    y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

    # normalize covarep and facet features, remove possible NaN values
    visual_max = np.max(np.max(np.abs(train_set_visual), axis=0), axis=0)
    visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize this dimension
    train_set_visual = train_set_visual / visual_max
    valid_set_visual = valid_set_visual / visual_max
    test_set_visual = test_set_visual / visual_max

    train_set_visual[train_set_visual != train_set_visual] = 0
    valid_set_visual[valid_set_visual != valid_set_visual] = 0
    test_set_visual[test_set_visual != test_set_visual] = 0

    audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
    train_set_audio = train_set_audio / audio_max
    valid_set_audio = valid_set_audio / audio_max
    test_set_audio = test_set_audio / audio_max

    train_set_audio[train_set_audio != train_set_audio] = 0
    valid_set_audio[valid_set_audio != valid_set_audio] = 0
    test_set_audio[test_set_audio != test_set_audio] = 0

    if mode == "all":
        x_train = np.concatenate((train_set_visual, train_set_audio, train_set_text), axis=2)
        x_valid = np.concatenate((valid_set_visual, valid_set_audio, valid_set_text), axis=2)
        x_test = np.concatenate((test_set_visual, test_set_audio, test_set_text), axis=2)
    if mode == "AV":
        x_train = np.concatenate((train_set_visual, train_set_audio), axis=2)
        x_valid = np.concatenate((valid_set_visual, valid_set_audio), axis=2)
        x_test = np.concatenate((test_set_visual, test_set_audio), axis=2)
    if mode == "AT":
        x_train = np.concatenate((train_set_audio, train_set_text), axis=2)
        x_valid = np.concatenate((valid_set_audio, valid_set_text), axis=2)
        x_test = np.concatenate((test_set_audio, test_set_text), axis=2)
    if mode == "VT":
        x_train = np.concatenate((train_set_visual, train_set_text), axis=2)
        x_valid = np.concatenate((valid_set_visual, valid_set_text), axis=2)
        x_test = np.concatenate((test_set_visual, test_set_text), axis=2)


    model = Sequential()

    if n_layers == 1:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(LSTM(64,return_sequences=True,input_shape=(max_len, x_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))        
    if n_layers == 2:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(LSTM(64,return_sequences=True,input_shape=(max_len, x_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(64,return_sequences=True,input_shape=(max_len, x_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))        
    if n_layers == 3:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(LSTM(64,return_sequences=True,input_shape=(max_len, x_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(64,return_sequences=True,input_shape=(max_len, x_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(64,input_shape=(max_len, x_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))        
    # you can try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)  # generate experiment dir
    filepath = "{}/best_validation_{}".format(saved_models_filepath, experiment_name) + ".ckpt"
    
    tensor_board = TensorBoard(log_dir=logs_filepath, histogram_freq=0, batch_size=batch_size, write_graph=True, 
        write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger('early_val.log')
    callbacks_list = [checkpoint, csv_logger, tensor_board]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_valid, y_valid],
              callbacks=callbacks_list)
    preds = model.predict(x_test)
    acc = np.mean((preds > 0.5) == y_test.reshape(-1, 1))

    print("batch_size="+str(batch_size))
    print("batch_norm="+str(batch_norm))
    print("dropout_rate="+str(dropout_rate))
    print("n_layers="+str(n_layers))
    print("max_len="+str(max_len))
    print("epochs="+str(epochs))
    print("mode="+str(mode))


    print("accuracy="+str(acc))
    model.load_weights(filepath)
    preds = model.predict(x_test)
    acc = np.mean((preds > 0.5) == y_test.reshape(-1, 1))
    print("best_valid_accuracy="+str(acc))
   



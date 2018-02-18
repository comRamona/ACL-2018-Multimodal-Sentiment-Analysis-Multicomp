#!/usr/bin/env python

'''
This script shows you how to:
1. Load the features for each segment;
2. Load the labels for each segment;
3. Perform word level feature alignment;
3. Use Keras to implement a simple LSTM on top of the data
'''

from __future__ import print_function
import numpy as np
import pandas as pd
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, BatchNormalization
from mmdata import Dataloader, Dataset


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
    mosi = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI')
    covarep = mosi.covarep()
    sentiments = mosi.sentiments() # sentiment labels, real-valued. for this tutorial we'll binarize them
    train_ids = mosi.train() # set of video ids in the training set
    test_ids = mosi.valid() # set of video ids in the valid set
#    test_ids = mosi.test() # set of video ids in the test set


    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
        for sid in covarep['covarep'][vid].keys():
            train_set_ids.append((vid, sid))

    test_set_ids = []
    for vid in test_ids:
        for sid in covarep['covarep'][vid].keys():
            test_set_ids.append((vid, sid))


    # partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
    # data will have shape (dataset_size, max_len, feature_dim)
    max_len = 15

    train_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if covarep['covarep'][vid][sid]], axis=0)
#    valid_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['covarep'][vid][sid]], axis=0)
    test_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if covarep['covarep'][vid][sid]], axis=0)

    # binarize the sentiment scores for binary classification task
    y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
    y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

    # normalize covarep and facet features, remove possible NaN values
    audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
    train_set_audio = train_set_audio / audio_max
    valid_set_audio = valid_set_audio / audio_max
    test_set_audio = test_set_audio / audio_max

    train_set_audio[train_set_audio != train_set_audio] = 0
    valid_set_audio[valid_set_audio != valid_set_audio] = 0
    test_set_audio[test_set_audio != test_set_audio] = 0

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


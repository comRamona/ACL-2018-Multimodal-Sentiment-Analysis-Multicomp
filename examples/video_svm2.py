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
from sklearn.svm import SVR,SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
import scipy
import scipy.stats

import sys


target_names = ['strg_neg', 'weak_neg', 'neutral', 'weak_pos', 'strg_pos']

def pad(data, max_len):
    """A funtion for padding/truncating sequence data to a given lenght"""
    # recall that data at each time step is a tuple (start_time, end_time, feature_vector), we only take the vector
    data = np.array([feature[2] for feature in data])
    n_rows = data.shape[0]
    dim = data.shape[1]
#    print(data.shape)
    if max_len >= n_rows:
        diff = max_len - n_rows
        padding = np.zeros((diff, dim))
        padded = np.concatenate((padding, data))
        return padded
    else:
        return np.concatenate(data[-max_len:])

if __name__ == "__main__":
    # Download the data if not present
    mosi = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSI')
    facet = mosi.facet()
    sentiments = mosi.sentiments() # sentiment labels, real-valued. for this tutorial we'll binarize them
    train_ids = mosi.train() # set of video ids in the training set
    valid_ids = mosi.valid() # set of video ids in the valid set
    test_ids = mosi.test() # set of video ids in the test set


    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
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


    # partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
    # data will have shape (dataset_size, max_len, feature_dim)
    max_len = 20

    train_set_video = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids if facet['facet'][vid][sid]], axis=0)
    valid_set_video = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids if facet['facet'][vid][sid]], axis=0)
    test_set_video = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids if facet['facet'][vid][sid]], axis=0)

    # binarize the sentiment scores for binary classification task
    y_train_bin = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
    y_valid_bin = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0
    y_test_bin = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

    # for regression
    y_train_reg = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
    y_valid_reg = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
    y_test_reg = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])

    train_set_video[train_set_video != train_set_video] = 0
    valid_set_video[valid_set_video != valid_set_video] = 0
    test_set_video[test_set_video != test_set_video] = 0

    x_train = train_set_video
    x_valid = valid_set_video
    x_test = test_set_video

    # create and train SVM for binary - Classification
    clf = SVC(kernel="linear")
    trained_model = clf.fit(x_train, y_train_bin)
    predictions = clf.predict(x_valid)
    acc = accuracy_score(y_valid_bin, predictions)
    print("Binary")
    print(classification_report(y_valid_bin, predictions, target_names=target_names))
    print("accuracy: "+str(acc))

    # create and train and SVM for continous - Regression
    clf = SVR(C=1.0, epsilon=0.2)
    trained_model = clf.fit(x_train, y_train_reg) 
    predictions = clf.predict(x_valid)
    mae = mean_absolute_error(y_valid_reg, predictions)  
    pr = scipy.stats.pearsonr(y_valid_reg,predictions)
    print("Regression")
    print("mae: " + str(mae))
    print("pr: " + str(pr))

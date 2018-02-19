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
f_limit = 36

def norm(data, max_len):
    # recall that data at each time step is a tuple (start_time, end_time, feature_vector), we only take the vector
    data = np.array([feature[2] for feature in data])
    data = data[:,:f_limit]
    n_rows = data.shape[0]
    dim = data.shape[1]
#    print("dims: ",max_len,dim,n_rows)
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    var = np.var(data,axis=0)
#    print("mean: "+str(mean.shape))
#    print("std: "+str(std.shape))
#    print("var: "+str(var.shape))
    res = np.concatenate((mean, std, var),axis=0)
#    print("all: ",res.shape)
    return mean


def multiclass(data):
    new_data = []
    for item in data:
        if item <= -1.8:
            new_data.append(0)
        elif item <= -0.6:
            new_data.append(1)
        elif item <= 0.6:
            new_data.append(2)
        elif item <= 1.8:
            new_data.append(3)
        elif item <= 3.0:
            new_data.append(4)
    return new_data



if __name__ == "__main__":
    # Download the data if not present
    mosi = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSI')
    covarep = mosi.covarep()
    sentiments = mosi.sentiments() # sentiment labels, real-valued. for this tutorial we'll binarize them
    train_ids = mosi.train() # set of video ids in the training set
    valid_ids = mosi.valid() # set of video ids in the valid set
    test_ids = mosi.test() # set of video ids in the test set


    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
        for sid in covarep['covarep'][vid].keys():
            train_set_ids.append((vid, sid))

    valid_set_ids = []
    for vid in valid_ids:
        for sid in covarep['covarep'][vid].keys():
                valid_set_ids.append((vid, sid))

    test_set_ids = []
    for vid in test_ids:
        for sid in covarep['covarep'][vid].keys():
            test_set_ids.append((vid, sid))


    # partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
    # data will have shape (dataset_size, max_len, feature_dim)
    max_len = 15

    train_set_audio = np.array([norm(covarep['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if covarep['covarep'][vid][sid]])
    valid_set_audio = np.array([norm(covarep['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if covarep['covarep'][vid][sid]])
    test_set_audio = np.array([norm(covarep['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if covarep['covarep'][vid][sid]])



#    train_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if covarep['covarep'][vid][sid]], axis=0)
#    valid_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if covarep['covarep'][vid][sid]], axis=0)
#    test_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if covarep['covarep'][vid][sid]], axis=0)

    # binarize the sentiment scores for binary classification task
    y_train_bin = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
    y_valid_bin = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0
    y_test_bin = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

    # for regression
    y_train_reg = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
    y_valid_reg = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
    y_test_reg = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])

    # for multiclass
    y_train_mc = multiclass(np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]))
    y_valid_mc = multiclass(np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]))
    y_test_mc = multiclass(np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]))


    # normalize covarep and facet features, remove possible NaN values
    audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
    train_set_audio = train_set_audio / audio_max
    valid_set_audio = valid_set_audio / audio_max
    test_set_audio = test_set_audio / audio_max

    train_set_audio[train_set_audio != train_set_audio] = 0
    valid_set_audio[valid_set_audio != valid_set_audio] = 0
    test_set_audio[test_set_audio != test_set_audio] = 0

    x_train = train_set_audio
    x_valid = valid_set_audio
    x_test = test_set_audio

    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)


    # create and train SVM for binary - Classification
    clf = SVC(kernel="linear")
    trained_model = clf.fit(x_train, y_train_bin)
    predictions = clf.predict(x_test)
    acc = accuracy_score(y_test_bin, predictions)
    print("Binary")
    print(classification_report(y_test_bin, predictions))
    print("accuracy: "+str(acc))


    clf2 = SVC(kernel="rbf")
    trained_model = clf2.fit(x_train, y_train_bin)
    predictions = clf2.predict(x_test)
    acc = accuracy_score(y_test_bin, predictions)
    print("Binary")
    print(classification_report(y_test_bin, predictions))
    print("accuracy: "+str(acc))


    clf3 = SVC(kernel="poly")
    trained_model = clf3.fit(x_train, y_train_bin)
    predictions = clf3.predict(x_test)
    acc = accuracy_score(y_test_bin, predictions)
    print("Binary")
    print(classification_report(y_test_bin, predictions))
    print("accuracy: "+str(acc))

    sys.exit()


    # create and train SVM for 5-class - Classification
    clf = OneVsRestClassifier(SVC(kernel="poly"))
    trained_model = clf.fit(x_train, y_train_mc)
    predictions = clf.predict(x_valid)
    acc = accuracy_score(y_valid_mc, predictions)
    print("5-class")
    print(classification_report(y_valid_mc, predictions, target_names=target_names, digits=5))
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

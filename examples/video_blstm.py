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
from keras.layers import Dense, Dropout, Embedding, LSTM, BatchNormalization,Bidirectional
from mmdata import Dataloader, Dataset
from keras.optimizers import SGD

target_names = ['strg_neg', 'weak_neg', 'neutral', 'weak_pos', 'strg_pos']
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
    return res


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
    max_len = 15

#use when norming via mean, var, std
    train_set_video = np.array([norm(facet['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids if facet['facet'][vid][sid]]) 
    valid_set_video = np.array([norm(facet['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids if facet['facet'][vid][sid]])      
    test_set_video = np.array([norm(facet['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids if facet['facet'][vid][sid]]) 

    # binarize the sentiment scores for binary classification task
    y_train_bin = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
    y_valid_bin = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0
    y_test_bin = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

    # for regression
    y_train_reg = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
    y_valid_reg = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
    y_test_reg = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])

    # for multiclass
#    y_train_mc = multiclass(np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]))
#    y_valid_mc = multiclass(np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]))
#    y_test_mc = multiclass(np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]))


    train_set_video[train_set_video != train_set_video] = 0
    valid_set_video[valid_set_video != valid_set_video] = 0
    test_set_video[test_set_video != test_set_video] = 0

    x_train = train_set_video
    x_valid = valid_set_video
    x_test = test_set_video

    lr = 0.01
    momentum = 0.9
    batch_size = 32
    train_epoch = 20
    sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
    optimizer = {'sgd': sgd}

    f_Facet_num = x_train.shape[1]
    model = Sequential()
    model.add(Bidirectional(LSTM(64),input_shape=(f_Facet_num,), merge_mode="concat"))
    model.add(Bidirectional(LSTM(64)))
    model.add(Activation('sigmoid'))

    # you can try using different optimizers and different optimizer configs
    model.compile(optimizer=optimizer[opt], loss='mae')
    model.fit(x_train, y_train_reg, validation_data=(x_valid,y_valid_reg), epochs=train_epoch, batch_size=batch_size)

    predictions = model.predict(x_test, verbose=0)
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

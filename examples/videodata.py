#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from mmdata import MOSI
from sklearn.preprocessing import label_binarize

class VideoData():

    def __init__(self, dataset=None):

        if dataset==None:
            self.dataset = MOSI()
        else:
            self.dataset = dataset
        self.train_ids = self.dataset.train()
        self.valid_ids = self.dataset.valid()
        self.test_ids = self.dataset.test()
        self.sentiments = self.dataset.sentiments()

    def pad(data, max_len):
    	"""A function for padding/truncating sequence data to a given lenght"""
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

    def get_data(self, data, max_len):

        x_train = []
        y_train = []
        x_test =[]
        y_test = []
        x_val = []
        y_val = []
        for vid, vdata in data.items(): # note that even Dataset with one feature will require explicit indexing of features
            for sid, sdata in vdata.items():           
                if sdata == []:
                    continue
                example = []
                for i, time_step in enumerate(sdata):
                    # data is truncated for 15 words
                    if i == max_len:
                        break
                    example.append(time_step[2]) # here first 2 dims (timestamps) will not be used

                for i in range(max_len - len(sdata)):
                    example.append(np.zeros(sdata[0][2].shape)) # padding each example to max_len
                example = np.asarray(example)
                label = 1 if self.sentiments[vid][sid] >= 0 else 0 # binarize the labels
                # here we just use everything except training set as the test set
                if vid in self.train_ids:
                    x_train.append(example)
                    y_train.append(label)
                elif vid in self.valid_ids:
                    x_val.append(example)
                    y_val.append(label)
                elif vid in self.test_ids:
                    x_test.append(example)
                    y_test.append(label)
        # Prepare the final inputs as numpy arrays
        x_train = np.asarray(x_train)
        x_val = np.asarray(x_val)
        x_test = np.asarray(x_test)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        y_test = np.asarray(y_test)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def get_video(self ,max_len=20):

        facet = self.dataset.facet()
        train_set_video, valid_set_video, test_set_video, y_train, y_val, y_test = self.get_data(facet["facet"], max_len)
        #train_set_video = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids if facet['facet'][vid][sid]], axis=0)
        #valid_set_video = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids if facet['facet'][vid][sid]], axis=0)
        #test_set_video = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids if facet['facet'][vid][sid]], axis=0)
        video_max = np.max(np.max(np.abs(train_set_video), axis=0), axis=0)
        video_max[video_max==0] = 1
        train_set_video = train_set_video / video_max
        valid_set_video = valid_set_video / video_max
        test_set_video = test_set_video / video_max
        train_set_video[train_set_video != train_set_video] = 0
        valid_set_video[valid_set_video != valid_set_video] = 0
        test_set_video[test_set_video != test_set_video] = 0

        return train_set_video, valid_set_video, test_set_video, y_train, y_val, y_test

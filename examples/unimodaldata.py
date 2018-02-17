#!/usr/bin/env python

'''
This script shows you how to:
1. Load the features for each segment;
2. Load the labels for each segment;
3. Prerocess data and use Keras to implement a simple LSTM on top of the data
'''
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from mmdata import MOSI
from sklearn.preprocessing import label_binarize

class UnimodalData():

    def __init__(self, dataset=None):

        if dataset==None:
            self.dataset = MOSI()
        else:
            self.dataset = dataset
        self.train_ids = self.dataset.train()
        self.valid_ids = self.dataset.valid()
        self.test_ids = self.dataset.test()
        self. sentiments = self.dataset.sentiments()

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

    def get_text(self, max_len=20):

        embeddings = self.dataset.embeddings()
        return self.get_data(embeddings["embeddings"], max_len)

    def get_words(self):

        words = self.dataset.words()
        x_train = []
        y_train = []
        x_test =[]
        y_test = []
        x_val = []
        y_val = []
        for vid, vdata in words["words"].items(): # note that even Dataset with one feature will require explicit indexing of features
            for sid, sdata in vdata.items():           
                if sdata == []:
                    continue
                example = []
                for i, time_step in enumerate(sdata):
                    example.append(time_step[2]) 
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


    def get_audio(self, max_len=20):

        covarep = self.dataset.covarep()
        train_set_audio, valid_set_audio, test_set_audio, y_train, y_val, y_test = self.get_data(covarep["covarep"], max_len)

        audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
        audio_max[audio_max==0] = 1
        train_set_audio = train_set_audio / audio_max
        valid_set_audio = valid_set_audio / audio_max
        test_set_audio = test_set_audio / audio_max

        train_set_audio[train_set_audio != train_set_audio] = 0
        valid_set_audio[valid_set_audio != valid_set_audio] = 0
        test_set_audio[test_set_audio != test_set_audio] = 0

        return train_set_audio, valid_set_audio, test_set_audio, y_train, y_val, y_test

    def get_video(self ,max_len=20):

        facet = self.dataset.facet()
        train_set_visual, valid_set_visual, test_set_visual, y_train, y_val, y_test = self.get_data(facet["facet"], max_len)
        visual_max = np.max(np.max(np.abs(train_set_visual), axis=0), axis=0)
        visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize this dimension
        train_set_visual = train_set_visual / visual_max
        valid_set_visual = valid_set_visual / visual_max
        test_set_visual = test_set_visual / visual_max
        train_set_visual[train_set_visual != train_set_visual] = 0
        valid_set_visual[valid_set_visual != valid_set_visual] = 0
        test_set_visual[test_set_visual != test_set_visual] = 0

        return train_set_visual, valid_set_visual, test_set_visual


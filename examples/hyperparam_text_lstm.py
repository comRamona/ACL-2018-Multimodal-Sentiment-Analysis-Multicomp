#!/usr/bin/env python

'''
This script shows you how to:
1. Load the features for each segment;
2. Load the labels for each segment;
3. Prerocess data and use Keras to implement a simple LSTM on top of the data
'''
from hyperas.distributions import uniform, choice, conditional
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from mmdata import MOSI
from utils.storage import build_experiment_folder, save_statistics

import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(0)

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    mosei = MOSI()
    embeddings = mosei.embeddings()
    sentiments = mosei.sentiments()
    train_ids = mosei.train()
    valid_ids = mosei.valid()
    #test_ids = mosei.test()

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    max_len = 15
    for vid, vdata in embeddings['embeddings'].items(): # note that even Dataset with one feature will require explicit indexing of features
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
            label = 1 if sentiments[vid][sid] >= 0 else 0 # binarize the labels

            # here we just use everything except training set as the test set
            if vid in train_ids:
                x_train.append(example)
                y_train.append(label)
            elif vid in valid_ids:
                x_test.append(example)
                y_test.append(label)

    # Prepare the final inputs as numpy arrays
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    
    model = Sequential()
    

    choiceval = {{choice(["one", "two", "three"])}}
    if conditional(choiceval) == "one":
        model.add(LSTM({{choice([64,128])}}, input_shape=(15, 300)))
        model.add(Dropout({{choice([0.2, 0.5])}}))
    elif conditional(choiceval) == "two":
        model.add(LSTM({{choice([64,128])}}, return_sequences=True,
               input_shape=(max_len, 300)))
        model.add(Dropout({{choice([0.2, 0.5])}}))
        model.add(LSTM({{choice([64,128])}}))
    elif conditional(choiceval) == "three":
        model.add(LSTM({{choice([64,128])}}, return_sequences=True,
               input_shape=(max_len, 300)))
        model.add(Dropout({{choice([0.2, 0.5])}}))
        model.add(LSTM({{choice([64,128])}}, return_sequences=True))
        model.add(Dropout({{choice([0.2, 0.5])}}))
        model.add(LSTM({{choice([64,128])}}))

    model.add(Dropout({{choice([0.2, 0.5])}}))   
    model.add(Dense(100, activation="relu"))
    model.add(Dropout({{choice([0.2, 0.5])}}))
    model.add(Dense(1, activation='sigmoid'))

    # adam = Adam(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
    # rmsprop = RMSprop(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
    # sgd = SGD(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
 
    # choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    # if conditional(choiceval) == 'adam':
    #     optim = adam
    # elif conditional(choiceval) == 'rmsprop':
    #     optim = rmsprop
    # else:
    #     optim = sgd

    optim = Adam(lr=0.0001)

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=optim)

    model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=50,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
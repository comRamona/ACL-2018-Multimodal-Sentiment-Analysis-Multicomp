#!/usr/bin/env python


######### SEEDS

import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
seed = 16122017
np.random.seed(seed)
rn.seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(seed)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
################################################################

import argparse
from collections import defaultdict
from mmdata import MOSI
from utils.parser_utils import KerasParserClass
from utils.storage import build_experiment_folder, save_statistics

parser = argparse.ArgumentParser(description='Welcome to LSTM experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, batch_norm, \
experiment_prefix, dropout_rate, n_layers, max_len = parser_extractor.get_argument_variables()


experiment_name = "experiment_{}_batch_size_{}_bn_{}_dr{}_nl_{}_ml_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   dropout_rate, n_layers, max_len)
max_len = 15

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.regularizers import l2
from keras.regularizers import l1
from keras.layers import Input, LSTM, Dense, TimeDistributed, Masking, Dropout, Bidirectional
from keras.models import Model
from unimodaldata import UnimodalData
from concrete import ConcreteDropout

m = UnimodalData()
x_train, x_val, x_test, y_train, y_val, y_test = m.get_text(max_len)
  
print("Data preprocessing finished! Begin compiling and training model.")
print(x_train.shape)
print(x_val.shape)


units = 64
model = Sequential()
# if n_layers == 1:
#     model.add(Bidirectional(LSTM(units), input_shape=(max_len, 300)))
#     model.add(Dropout(dropout_rate))

# else:
#     model.add(LSTM(units, return_sequences=True,
#                input_shape=(max_len, 300))) 
#     if n_layers > 2:
#         for i in range(n_layers - 2):
#             model.add(Dropout(dropout_rate))
#             model.add(LSTM(units, return_sequences=True))
#     model.add(Dropout(dropout_rate))
#     model.add(LSTM(units))
 
model.add(Bidirectional(LSTM(units), input_shape=(max_len, 300)))
model.add(Dropout(dropout_rate))
model.add(Dense(200, activation="relu", W_regularizer=l2(0.0001)))
model.add(Dropout(dropout_rate))
model.add(Dense(200, activation="relu", W_regularizer=l2(0.0001)))
model.add(Dropout(dropout_rate))
model.add(Dense(200, activation="relu", W_regularizer=l2(0.0001)))
model.add(Dense(1, activation='sigmoid'))


saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)  # generate experiment dir


# try using different optimizers and different optimizer configs
optimizer = Adam(lr=0.0001)
model.compile("adadelta", 'binary_crossentropy', metrics=['accuracy'])
filepath = "{}/best_validation_{}".format(saved_models_filepath, experiment_name) + ".hdfs5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=10,
                              verbose=1, mode='max')
tensor_board = TensorBoard(log_dir=logs_filepath, histogram_freq=0, batch_size=batch_size, write_graph=True, 
    write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
callbacks_list = [checkpoint, early_stopping, tensor_board]
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=50,
          validation_data=[x_val, y_val],
          callbacks=callbacks_list)

model.load_weights(filepath)
score, acc = model.evaluate(x_test, y_test)
print("Test accuracy with no dropout:",acc)

f = K.function([model.layers[0].input, K.learning_phase()],
                      [model.layers[-1].output])

def predict_with_uncertainty(f, x, no_classes, n_iter=100):
    """
    Dropout as a bayesian approximation
    
    :param x: x_test / x_validation to make predictions over
    :param f: forward propagation function with dropout turned on
    :param no_classes: number of classes
    :param n_iter: number of montecarlo samples
    
    :output prediction: Montecarlo Dropout estimate of mean prediction
    :output uncertainty: Montecarlo Dropout estimate of error bars (standard deviation)
    """
    mu = np.zeros( (x.shape[0], no_classes) )
    sq = np.zeros( (x.shape[0], no_classes) )

    for i in range(n_iter):
        nn = f((x, 1))[0]
        mu += nn
        sq += nn**2

    prediction = mu / n_iter
    uncertainty =  np.sqrt(nn/n_iter -  prediction**2)
    return prediction, uncertainty   

pred, uncertainty = predict_with_uncertainty(f,x_test,500)
test_acc = np.mean((pred > 0.5) == y_test.reshape(-1, 1))
print("Test accuracy with bayesian dropout: {}".format(test_acc))
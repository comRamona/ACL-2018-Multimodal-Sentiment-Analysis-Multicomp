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

from collections import defaultdict
from mmdata import MOSI
import argparse
from collections import defaultdict
from mmdata.dataset import Dataset
import time

from utils.parser_utils import KerasParserClass
from utils.storage import build_experiment_folder, save_statistics
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM
from keras.layers import Merge
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras import optimizers
from keras.regularizers import l2
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization
from keras.constraints import nonneg
from sklearn.metrics import mean_absolute_error
from keras.optimizers import SGD
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.regularizers import Regularizer

class CenterL1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0., c=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.c = K.cast_to_floatx(c)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x - self.c))
        if self.l2:
            regularization += K.sum(0.001 * K.square(x - self.c))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


# Aliases.
def centerl2(l, c):
    return CenterL1L2(l2=l, c=c)


parser = argparse.ArgumentParser(description='Welcome to LSTM experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, batch_norm, \
experiment_prefix, dropout_rate, n_layers, max_len = parser_extractor.get_argument_variables()

batch_size = 64
seed = 0
epochs = 50
dropout_rate = 0.1

if(experiment_prefix =="classification"):
    experiment_prefix = "Trimodal" + time.strftime("%Y-%m-%d %H:%M")
experiment_name = "experiment_{}_batch_size_{}_bn_{}_dr{}_nl_{}_ml_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   dropout_rate, n_layers, max_len)
saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)
filepath = "{}/best_validation_{}".format(saved_models_filepath, experiment_name)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

from multimodaldata import get_data
train_set_audio, valid_set_audio, test_set_audio, train_set_text, valid_set_text, test_set_text, \
train_set_visual, valid_set_visual, test_set_visual, \
y_train, y_valid, y_test = get_data(max_len_audio=20, max_len_text=20, max_len_visual=20)


filepath = "{}/best_validation_{}_".format(saved_models_filepath, experiment_name)
weights = "{}_text_weights{}.h5"

checkpoint1 = ModelCheckpoint(weights.format(filepath,2), monitor='val_acc',
save_best_only=True, verbose=1, mode="auto")
early_stopping1 = EarlyStopping(monitor="val_acc", patience=10, mode="max")
model = Sequential()
model.add(Dense(46, input_shape=(train_set_text.shape[1], train_set_text.shape[2])))
model.add(LSTM(64))
model.add(Dense(100, activation="relu")) 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_set_text, y=y_train, batch_size=64, epochs=epochs,
             verbose=1, validation_data=[valid_set_text, y_valid], shuffle=True, callbacks = [checkpoint1, early_stopping1])
model.load_weights(weights.format(filepath,2))

preds = model.predict(test_set_text)
acc = np.mean((preds > 0.5) == y_test.reshape(-1, 1))
print("Test accuracy text: ", acc)

layer1_weights = model.layers[1].get_weights()[0]


weights = "{}_visualp_weights{}.h5"

checkpoint2 = ModelCheckpoint(weights.format(filepath,2), monitor='val_acc',
save_best_only=True, verbose=1, mode="auto")
model2 = Sequential()
model2.add(LSTM(64, input_shape=(train_set_visual.shape[1], train_set_visual.shape[2]), W_regularizer = centerl2(0.01, layer1_weights)))  
model2.add(Dense(100, activation="relu", W_regularizer = centerl2(0.01, model.layers[2].get_weights()[0])))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model2.fit(train_set_visual, y=y_train, batch_size=64, epochs=epochs,
             verbose=1, validation_data=[valid_set_visual, y_valid], shuffle=True, callbacks = [checkpoint2, early_stopping1])

model2.load_weights(weights.format(filepath,2))
preds = model2.predict(test_set_visual)
acc = np.mean((preds > 0.5) == y_test.reshape(-1, 1))
print("Test accuracy visual, priors on text weights: ", acc)


weights = "{}_visual_weights{}.h5"

checkpoint3 = ModelCheckpoint(weights.format(filepath,2), monitor='val_acc',
save_best_only=True, verbose=1, mode="auto")
model3 = Sequential()
model3.add(LSTM(64, input_shape=(train_set_visual.shape[1], train_set_visual.shape[2])))
model3.add(Dense(100, activation="relu"))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.summary()
model3.fit(train_set_visual, y=y_train, batch_size=64, epochs=epochs,
             verbose=1, validation_data=[valid_set_visual, y_valid], shuffle=True, callbacks = [checkpoint3, early_stopping1])
model3.load_weights(weights.format(filepath,2))
preds = model3.predict(test_set_visual)
acc = np.mean((preds > 0.5) == y_test.reshape(-1, 1))
print("Test accuracy visual, no priors: ", acc)
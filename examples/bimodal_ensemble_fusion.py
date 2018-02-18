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

from collections import defaultdict
from mmdata import MOSI
import argparse
from collections import defaultdict
from mmdata.dataset import Dataset
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
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

seed = 16122017
np.random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
tf.set_random_seed(seed)

parser = argparse.ArgumentParser(description='Welcome to LSTM experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input
batch_size, seed, epochs, logs_path, continue_from_epoch, batch_norm, \
experiment_prefix, dropout_rate, n_layers, max_len = parser_extractor.get_argument_variables()


experiment_name = "experiment_{}_batch_size_{}_bn_{}_dr{}_nl_{}_ml_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   dropout_rate, n_layers, max_len)
from multimodaldata import get_data
train_set_audio, valid_set_audio, test_set_audio, train_set_text, valid_set_text, test_set_text, \
train_set_visual, valid_set_visual, test_set_visual, \
y_train, y_valid, y_test = get_data(max_len_audio=20, max_len_text=15, max_len_visual=20)

# pdb.set_trace()
k=3
m=2
# AUDIO
model1_in = Input(name="Input1",shape=(train_set_audio.shape[1], train_set_audio.shape[2]))
model1_cnn = Conv1D(filters=64, kernel_size=k, activation='relu')(model1_in)
model1_mp = MaxPooling1D(m)(model1_cnn)
model1_fl = Flatten()(model1_mp)
model1_dense = Dense(128, activation="relu", W_regularizer=l2(0.0001))(model1_fl)
model1_out = Dense(1, name='layer_1')(model1_dense)
model1 = Model(model1_in, model1_out)
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()
checkpoint1 = ModelCheckpoint('b_weights1.h5', monitor='val_acc',
save_best_only=True, verbose=2, mode="max")
early_stopping1 = EarlyStopping(monitor="val_acc", patience=20, mode="max")
model1.fit(train_set_audio, y=y_train, batch_size=50, epochs=100,
             verbose=1, validation_data=[valid_set_audio, y_valid], shuffle=True, callbacks=[early_stopping1, checkpoint1])
model1.load_weights("b_weights1.h5")
score, acc = model1.evaluate(test_set_audio, y_test)
print("Audio Test accuracy: ", acc)

# TEXT
model2_in = Input(name="Input2",shape=(train_set_text.shape[1], train_set_text.shape[2]))
model2_blstm = Bidirectional(LSTM(64))(model2_in)
model2_d1 = Dropout(dropout_rate)(model2_blstm)
model2_d2 = Dense(200, activation="relu", W_regularizer=l2(0.0001))(model2_d1)
model2_d3 = Dropout(dropout_rate)(model2_d2)
model2_d4 = Dense(200, activation="relu", W_regularizer=l2(0.0001))(model2_d3)
model2_d5 = Dropout(dropout_rate)(model2_d4)
model2_d6 = Dense(200, activation="relu", W_regularizer=l2(0.0001))(model2_d5)
model2_out = Dense(1, name='layer_2')(model2_d6)
model2 = Model(model2_in, model2_out)
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()
checkpoint2 = ModelCheckpoint('b_weights2.h5', monitor='val_acc',
save_best_only=True, verbose=2)
early_stopping2 = EarlyStopping(monitor="val_acc", patience=20,mode="max")
model2.fit(train_set_text, y=y_train, batch_size=50, epochs=100,
             verbose=1, validation_data=[valid_set_text, y_valid], shuffle=True, callbacks=[early_stopping2, checkpoint2])
model2.load_weights("b_weights2.h5")
score, acc = model2.evaluate(test_set_text, y_test)
print("Text Test accuracy: ", acc)


concatenated = concatenate([model1_out, model2_out])
from keras.constraints import non_neg
out = Dense(1, activation='sigmoid', name='output_layer', kernel_constraint=non_neg(), bias_regularizer=l2(1))(concatenated)
merged_model = Model([model1_in, model2_in], out)
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
merged_model.summary()
checkpoint = ModelCheckpoint('b_weights.h5', monitor='val_acc',
save_best_only=True, verbose=2,mode="max")
early_stopping = EarlyStopping(monitor="val_acc", patience=20,mode="max")
merged_model.fit([train_set_audio, train_set_text], y=y_train, batch_size=50, epochs=100,
             verbose=1, validation_data=[[valid_set_audio, valid_set_text],y_valid], shuffle=True, 
callbacks=[early_stopping, checkpoint])

merged_model.load_weights("b_weights.h5")
score, acc = merged_model.evaluate([test_set_audio, test_set_text], y_test)
print("Test accuracy: ", acc)

model_json = merged_model.to_json()
with open("bimodal_model.json", "w") as json_file:
    json_file.write(model_json)
merged_model.save_weights("bimodal_model.h5")
print("Saved model to disk")

from keras.models import model_from_json
json_file = open('bimodal_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("bimodal_model.h5")
print("Loaded model from disk")
print(loaded_model.layers[-1].get_weights())
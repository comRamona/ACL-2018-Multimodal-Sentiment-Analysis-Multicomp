import pdb
import numpy as np
import pandas as pd
from collections import defaultdict
from mmdata import MOSI
import argparse
from collections import defaultdict
from mmdata.dataset import Dataset
seed = 16122017
np.random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
tf.set_random_seed(seed)

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

class Args:
    def __init__(self):
        self.units = 100
        self.l2 = 0.005
        self.lr = 0.01
        self.dropout = 0.3
        self.loss = 'binary_crossentropy'
        self.optimizer = 'adam'
        self.momentum = 0.1

args  = Args()
a, b=4.0, 8.0
train_patience = 10
parser = argparse.ArgumentParser(description='Welcome to LSTM experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, batch_norm, \
experiment_prefix, dropout_rate, n_layers, max_len = parser_extractor.get_argument_variables()


experiment_name = "experiment_{}_batch_size_{}_bn_{}_dr{}_nl_{}_ml_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   dropout_rate, n_layers, max_len)

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

max_len = 20
mosi = MOSI()
embeddings = mosi.embeddings()
facet = mosi.facet()
covarep = mosi.covarep()
sentiments = mosi.sentiments() # sentiment labels, real-valued. for this tutorial we'll binarize them
train_ids = mosi.train()
valid_ids = mosi.valid()
test_ids = mosi.test()
# sort through all the video ID, segment ID pairs
train_set_ids = []
for vid in train_ids:
    for sid in embeddings['embeddings'][vid].keys():
        if embeddings['embeddings'][vid][sid] and facet['facet'][vid][sid] and covarep['covarep'][vid][sid]:
            train_set_ids.append((vid, sid))

valid_set_ids = []
for vid in valid_ids:
    for sid in embeddings['embeddings'][vid].keys():
        if embeddings['embeddings'][vid][sid] and facet['facet'][vid][sid] and covarep['covarep'][vid][sid]:
            valid_set_ids.append((vid, sid))

test_set_ids = []
for vid in valid_ids:
    for sid in embeddings['embeddings'][vid].keys():
        if embeddings['embeddings'][vid][sid] and facet['facet'][vid][sid] and covarep['covarep'][vid][sid]:
            test_set_ids.append((vid, sid))

# partition the training, valid and tesembeddingsall sequences will be padded/truncated to 15 steps
# data will have shape (dataset_size, max_len, feature_dim)

train_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if covarep['covarep'][vid][sid]], axis=0)
valid_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if covarep['covarep'][vid][sid]], axis=0)
test_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if covarep['covarep'][vid][sid]], axis=0)

train_set_visual = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
valid_set_visual = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
test_set_visual = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

train_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
valid_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
test_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

train_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
valid_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
test_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)
# binarize the sentiment scores for binary classification task
y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0
y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

train_set_audio = train_set_audio[:,:,1:35]
valid_set_audio = valid_set_audio[:,:,1:35]
test_set_audio = test_set_audio[:,:,1:35]

visual_max = np.max(np.max(np.abs(train_set_visual), axis=0), axis=0)
visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize this dimension
train_set_visual = train_set_visual / visual_max
valid_set_visual = valid_set_visual / visual_max
test_set_visual = test_set_visual / visual_max

train_set_visual[train_set_visual != train_set_visual] = 0
valid_set_visual[valid_set_visual != valid_set_visual] = 0
test_set_visual[valid_set_visual != test_set_visual] = 0

audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
audio_max[audio_max==0] = 1
train_set_audio = train_set_audio / audio_max
valid_set_audio = valid_set_audio / audio_max
test_set_audio = test_set_audio / audio_max

train_set_audio[train_set_audio != train_set_audio] = 0
valid_set_audio[valid_set_audio != valid_set_audio] = 0
test_set_audio[valid_set_audio != test_set_audio] = 0

# pdb.set_trace()
k=3
m=2
model1_in = Input(name="Input1",shape=(train_set_audio.shape[1], train_set_audio.shape[2]))
model1_cnn = Conv1D(filters=64, kernel_size=k, activation='relu')(model1_in)
model1_mp = MaxPooling1D(m)(model1_cnn)
model1_fl = Flatten()(model1_mp)
model1_dense = Dense(128, activation="relu")(model1_fl)
model1_out = Dense(1, name='layer_1')(model1_dense)
model1 = Model(model1_in, model1_out)
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()
checkpoint1 = ModelCheckpoint('weights1.h5', monitor='val_acc',
save_best_only=True, verbose=2)
early_stopping1 = EarlyStopping(monitor="val_loss", patience=30)
model1.fit(train_set_audio, y=y_train, batch_size=50, epochs=100,
             verbose=1, validation_data=[valid_set_audio, y_valid], shuffle=True, callbacks=[early_stopping1, checkpoint1])
model1.load_weights("weights1.h5")


model2_in = Input(name="Input2",shape=(train_set_text.shape[1], train_set_text.shape[2]))
model2_cnn = Conv1D(filters=64, kernel_size=k, activation='relu')(model2_in)
model2_mp = MaxPooling1D(m)(model2_cnn)
model2_fl = Flatten()(model2_mp)
model2_dense = Dense(128, activation="relu")(model2_fl)
model2_out = Dense(1, name='layer_2')(model2_dense)
model2 = Model(model2_in, model2_out)
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()
checkpoint2 = ModelCheckpoint('weights2.h5', monitor='val_acc',
save_best_only=True, verbose=2)
early_stopping2 = EarlyStopping(monitor="val_loss", patience=30)
model2.fit(train_set_text, y=y_train, batch_size=50, epochs=100,
             verbose=1, validation_data=[valid_set_text, y_valid], shuffle=True, callbacks=[early_stopping2, checkpoint2])
model2.load_weights("weights2.h5")

concatenated = concatenate([model1_out, model2_out])
out = Dense(1, activation='sigmoid', name='output_layer')(concatenated)

merged_model = Model([model1_in, model2_in], out)
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
merged_model.summary()
checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc',
save_best_only=True, verbose=2)
early_stopping = EarlyStopping(monitor="val_loss", patience=30)
merged_model.fit([train_set_audio, train_set_text], y=y_train, batch_size=50, epochs=100,
             verbose=1, validation_data=[[valid_set_audio, valid_set_text],y_valid], shuffle=True, 
callbacks=[early_stopping, checkpoint])

merged_model.load_weights("weights.h5")
score, acc = merged_model.evaluate([test_set_audio, test_set_text], y_test)
print("Test accuracy: ", acc)
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

import logging
logging.basicConfig(filename="train-late-fusion.log", level=logging.INFO)
logging.root.level = logging.INFO

logging.info("New experiment")

parser = argparse.ArgumentParser(description='Welcome to LSTM experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input


batch_size = 64
seed = 1122017
epochs = 50
dropout_rate = 0.1
logs_path = "classification_logs/"
experiment_prefix = "late_fusion"
continue_from_epoch = -1
batch_norm = False
n_layers = 1
max_len = 20

experiment_name = "experiment_{}_batch_size_{}_bn_{}_dr{}_nl_{}_ml_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   dropout_rate, n_layers, max_len)
saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)
filepath = "{}/best_validation_{}".format(saved_models_filepath, experiment_name)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

from multimodaldata import get_data
train_set_audio, valid_set_audio, test_set_audio, train_set_text, valid_set_text, test_set_text, \
train_set_visual, valid_set_visual, test_set_visual, \
y_train, y_valid, y_test = get_data(max_len_audio=20, max_len_text=15, max_len_visual=20)


filepath = "{}/best_validation_{}_".format(saved_models_filepath, experiment_name)
weights = "{}late_fusion_weights{}.h5"
# pdb.set_trace()
k=3
m=2
# AUDIO
model1_in = Input(name="Audio_Covarep",shape=(train_set_audio.shape[1], train_set_audio.shape[2]))
model1_cnn = Conv1D(filters=64, kernel_size=k, activation='relu')(model1_in)
model1_mp = MaxPooling1D(m)(model1_cnn)
model1_fl = Flatten()(model1_mp)
model1_dense = Dense(128, activation="relu", W_regularizer=l2(0.0001))(model1_fl)

# TEXT
model2_in = Input(name="Text_GloVe",shape=(train_set_text.shape[1], train_set_text.shape[2]))
model2_blstm = Bidirectional(LSTM(64))(model2_in)
model2_d1 = Dropout(dropout_rate)(model2_blstm)
model2_d2 = Dense(200, activation="relu", W_regularizer=l2(0.0001))(model2_d1)
model2_d3 = Dropout(dropout_rate)(model2_d2)
model2_d4 = Dense(200, activation="relu", W_regularizer=l2(0.0001))(model2_d3)
model2_d5 = Dropout(dropout_rate)(model2_d4)
model2_d6 = Dense(200, activation="relu", W_regularizer=l2(0.0001))(model2_d5)

model3_in = Input(name="Video_Facet",shape=(train_set_visual.shape[1], train_set_visual.shape[2]))
model3_cnn = Conv1D(filters=64, kernel_size=k, activation='relu')(model3_in)
model3_mp = MaxPooling1D(m)(model3_cnn)
model3_fl = Flatten()(model3_mp)
model3_dense = Dense(128, activation="relu", W_regularizer=l2(0.001))(model3_fl)


concatenated = concatenate([model1_dense, model2_d6, model3_dense])
dense = Dense(200, activation='sigmoid', name="last_dense")(concatenated)
out = Dense(1, activation='sigmoid', name='Sigmoid_Output')(dense)

merged_model = Model([model1_in, model2_in, model3_in], out)
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
merged_model.summary()
csv_logger = CSVLogger('late_fusion_val.log')
tensor_board = TensorBoard(log_dir=logs_filepath, histogram_freq=0, batch_size=batch_size, write_graph=True, 
    write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
checkpoint = ModelCheckpoint(weights.format(filepath, "merged"), monitor='val_loss',
save_best_only=True, verbose=2, mode="min")
early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")


merged_model.fit([train_set_audio, train_set_text, train_set_visual], y=y_train, batch_size=batch_size, epochs=epochs,
             verbose=1, validation_data=[[valid_set_audio, valid_set_text, valid_set_visual],y_valid], shuffle=True, 
callbacks=[csv_logger, checkpoint, tensor_board, early_stopping])


merged_model.load_weights(weights.format(filepath, "merged"))
preds = merged_model.predict([test_set_audio, test_set_text, test_set_visual])
acc = np.mean((preds > 0.5) == y_test.reshape(-1, 1))
print("Test accuracy: ", acc)
print("TensorBoard: ", logs_filepath)
logging.info("Dropout rate: {} n_layers: {}".format(dropout_rate, n_layers))
logging.info("Test accuracy: " + str(acc))

model_json = merged_model.to_json()
with open(filepath + "late_fusion_model.json", "w") as json_file:
    json_file.write(model_json)
merged_model.save_weights(filepath + "late_fusion_model.h5")
print("Saved model to disk")

from keras.utils import plot_model
plot_model(merged_model, to_file='late_merged_modelr.png',show_shapes=True)
plot_model(merged_model, to_file='late_merged_modelrr.png',show_shapes=True, show_layer_names=False)
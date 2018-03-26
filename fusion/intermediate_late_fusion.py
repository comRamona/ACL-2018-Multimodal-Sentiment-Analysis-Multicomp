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
logging.basicConfig(filename="train-intermediate-late-fusion.log", level=logging.INFO)
logging.root.level = logging.INFO



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

experiment_name = "Late_fusion" + time.strftime("%Y-%m-%d %H:%M")
saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)
filepath = "{}/best_validation_{}".format(saved_models_filepath, experiment_name)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

from multimodaldata import get_data



filepath = "{}/best_validation_{}_".format(saved_models_filepath, experiment_name)
weights = "{}late_fusion_weights{}.h5"
# pdb.set_trace()
k=3
m=2

for max_len in [15, 20, 25, 30]:
    for dropout_rate in [0.0, 0.1, 0.2]:
      for n_layers in [1, 2, 3]:

          logging.info("New experiment")
          logging.info("*" * 30)

          train_set_audio, valid_set_audio, test_set_audio, train_set_text, valid_set_text, test_set_text, \
          train_set_visual, valid_set_visual, test_set_visual, \
          y_train, y_valid, y_test = get_data(max_len_audio=max_len, max_len_text=max_len, max_len_visual=max_len)

          # AUDIO
          model1_in = Input(name="Audio_Covarep",shape=(train_set_audio.shape[1], train_set_audio.shape[2]))
          model1_cnn = Conv1D(filters=64, kernel_size=k, activation='relu')(model1_in)
          model1_mp = MaxPooling1D(m)(model1_cnn)
          model1_fl = Flatten()(model1_mp)
          model1_dropout = Dropout(dropout_rate)(model1_fl)
          model1_dense = Dense(128, activation="relu", W_regularizer=l2(0.0001))(model1_dropout)
          for i in range(2, n_layers + 1):
            model1_dropout = Dropout(dropout_rate)(model1_dense)
            model1_dense = Dense(128, activation="relu", W_regularizer=l2(0.0001))(model1_dropout)

          # TEXT
          model2_in = Input(name="Text_GloVe",shape=(train_set_text.shape[1], train_set_text.shape[2]))
          model2_blstm = Bidirectional(LSTM(64))(model2_in)
          model2_dropout = Dropout(dropout_rate)(model2_blstm)
          model2_dense = Dense(128, activation="relu", W_regularizer=l2(0.0001))(model2_dropout)
          for i in range(2, n_layers + 1):
              model2_dropout = Dropout(dropout_rate)(model2_dense)
              model2_dense = Dense(128, activation="relu", W_regularizer=l2(0.0001))(model2_dropout)
         
          model3_in = Input(name="Video_Facet",shape=(train_set_visual.shape[1], train_set_visual.shape[2]))
          model3_cnn = Conv1D(filters=64, kernel_size=k, activation='relu')(model3_in)
          model3_mp = MaxPooling1D(m)(model3_cnn)
          model3_fl = Flatten()(model3_mp)
          model3_dropout = Dropout(dropout_rate)(model3_fl)
          model3_dense = Dense(128, activation="relu", W_regularizer=l2(0.0001))(model3_dropout)
          for i in range(2, n_layers + 1):
            model3_dropout = Dropout(dropout_rate)(model3_dense)
            model3_dense = Dense(128, activation="relu", W_regularizer=l2(0.0001))(model3_dropout)


          concatenated = concatenate([model1_dense, model2_dense, model3_dense])
          dense = Dense(200, activation='relu', name="mdense")(concatenated)
          dense2 = Dense(200, activation='relu', name="last_dense")(dense)
          out = Dense(1, activation='sigmoid', name='Sigmoid_Output')(dense2)

          merged_model = Model([model1_in, model2_in, model3_in], out)
          merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
          merged_model.summary()
          csv_logger = CSVLogger('late_fusion_val.log')
          tensor_board = TensorBoard(log_dir=logs_filepath, histogram_freq=0, batch_size=batch_size, write_graph=True, 
              write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
          checkpoint = ModelCheckpoint(weights.format(filepath, "merged"), monitor='val_acc',
          save_best_only=True, verbose=2, mode="max")
          early_stopping = EarlyStopping(monitor="val_acc", patience=10, mode="max")


          merged_model.fit([train_set_audio, train_set_text, train_set_visual], y=y_train, batch_size=batch_size, epochs=epochs,
                       verbose=1, validation_data=[[valid_set_audio, valid_set_text, valid_set_visual],y_valid], shuffle=True, 
          callbacks=[csv_logger, checkpoint, tensor_board, early_stopping])


          merged_model.load_weights(weights.format(filepath, "merged"))
          loss, acc = merged_model.evaluate([valid_set_audio, valid_set_text, valid_set_visual], y_valid)
          print("Dropout rate: {} n_layers: {} max_len: {}".format(dropout_rate, n_layers, max_len))
          logging.info("Dropout rate: {} n_layers: {}, max_len: {}".format(dropout_rate, n_layers, max_len))
          print("Validatin loss: {}, acc: {}".format(loss, acc))
          logging.info("Validatin loss: {}, acc: {}".format(loss, acc))

          loss, acc = merged_model.evaluate([test_set_audio, test_set_text, test_set_visual], y_test)
          print("Test set")
          logging.info("Test set:")
          print("Test loss: {}, acc: {}".format(loss, acc))
          logging.info("Test loss: {}, acc: {}".format(loss, acc))

          # model_json = merged_model.to_json()
          # with open(filepath + "late_fusion_model.json", "w") as json_file:
          #     json_file.write(model_json)
          # merged_model.save_weights(filepath + "late_fusion_model.h5")
          # print("Saved model as {}".format(filepath + "late_fusion_model.json"))
          # print("Saved weights as {}".format(filepath + "late_fusion_model.h5"))
          # logging.info("Saved model as {}".format(filepath + "late_fusion_model.json"))
          # logging.info("Saved weights as {}".format(filepath + "late_fusion_model.h5"))

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
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


parser = argparse.ArgumentParser(description='Welcome to LSTM experiments script')  # generates an argument parser
parser_extractor = KerasParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, batch_norm, \
experiment_prefix, dropout_rate, n_layers, max_len = parser_extractor.get_argument_variables()
if(experiment_prefix =="classification"):
    experiment_prefix = "Trimodal" + time.strftime("%Y-%m-%d %H:%M")
experiment_name = "experiment_{}_batch_size_{}_bn_{}_dr{}_nl_{}_ml_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   dropout_rate, n_layers, max_len)
saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)
filepath = "{}/best_validation_{}".format(saved_models_filepath, experiment_name)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

from multimodaldata_mm import get_data
train_set_audio, valid_set_audio, test_set_audio, train_set_text, valid_set_text, test_set_text, \
train_set_visual, valid_set_visual, test_set_visual, \
y_train, y_valid, y_test = get_data(max_len_audio=20, max_len_text=15, max_len_visual=20)


filepath = "{}/best_validation_{}_".format(saved_models_filepath, experiment_name)
weights = "{}_weights{}.h5"
# pdb.set_trace()
k=3
m=2

# VISUAL
model3_in = Input(name="Video_Facet",shape=(train_set_visual.shape[1], train_set_visual.shape[2]))
model3_cnn = Conv1D(filters=64, kernel_size=k, activation='relu')(model3_in)
model3_mp = MaxPooling1D(m)(model3_cnn)
model3_fl = Flatten()(model3_mp)
model3_dense = Dense(128, activation="relu", W_regularizer=l2(0.001))(model3_fl)
model3_out = Dense(1, name='Sigmoid_Video')(model3_dense)
model3 = Model(model3_in, model3_out)
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.summary()
checkpoint3 = ModelCheckpoint(weights.format(filepath,3), monitor='val_acc',
save_best_only=True, verbose=2, mode="max")
early_stopping3 = EarlyStopping(monitor="val_acc", patience=10, mode="max")
model3.fit(train_set_visual, y=y_train, batch_size=32, epochs=100,
             verbose=1, validation_data=[valid_set_visual, y_valid], shuffle=True, callbacks=[early_stopping3, checkpoint3])
model3.load_weights(weights.format(filepath,3))
preds = model3.predict(test_set_visual)
acc = np.mean((preds > 0.5) == y_test.reshape(-1, 1))
print("Video Test accuracy: ", acc)

concatenated = concatenate([model3_out])
out = Dense(1, activation='sigmoid', name='Sigmoid_Output')(concatenated)

merged_model = Model([model3_in], out)
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
merged_model.summary()

tensor_board = TensorBoard(log_dir=logs_filepath, histogram_freq=0, batch_size=batch_size, write_graph=True, 
    write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
checkpoint = ModelCheckpoint(weights.format(filepath, "merged"), monitor='val_acc',
save_best_only=True, verbose=2, mode="max")
early_stopping = EarlyStopping(monitor="val_acc", patience=15, mode="max")
merged_model.fit([train_set_audio, train_set_text, train_set_visual], y=y_train, batch_size=32, epochs=100,
             verbose=1, validation_data=[[valid_set_audio, valid_set_text, valid_set_visual],y_valid], shuffle=True, 
callbacks=[early_stopping, checkpoint, tensor_board])
merged_model.load_weights(weights.format(filepath, "merged"))

score, acc = merged_model.evaluate([test_set_audio, test_set_text, test_set_visual], y_test)
print("Test accuracy: ", acc)
print("TensorBoard: ", logs_filepath)

model_json = merged_model.to_json()
with open(filepath + "trimodal_model.json", "w") as json_file:
    json_file.write(model_json)
merged_model.save_weights(filepath + "trimodal_model.h5")
print("Saved model to disk")

from keras.models import model_from_json
json_file = open(filepath + 'trimodal_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(filepath + "trimodal_model.h5")
print("Loaded model from disk")
print(loaded_model.layers[-1].get_weights())

from keras.utils import plot_model
plot_model(merged_model, to_file='merged_model.png')
plot_model(model3, to_file='model3.png')

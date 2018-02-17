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
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.


# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
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

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

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

train_set_audio = np.mean(train_set_audio, axis=1)
valid_set_audio = np.mean(valid_set_audio, axis=1)
test_set_audio = np.mean(test_set_audio, axis=1)
train_set_visual = np.mean(train_set_visual, axis=1)
valid_set_visual = np.mean(valid_set_visual, axis=1)
test_set_visual = np.mean(test_set_visual, axis=1)

# pdb.set_trace()

end_to_end = True

Covarep_model = Sequential()
Covarep_model.add(BatchNormalization(input_shape=(train_set_audio.shape[1],), name = 'covarep_layer_0'))
Covarep_model.add(Dropout(0.2, name = 'covarep_layer_1'))
Covarep_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'covarep_layer_2', trainable=end_to_end))
Covarep_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'covarep_layer_3', trainable=end_to_end))
Covarep_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'covarep_layer_4', trainable=end_to_end))
Covarep_model.summary()

Facet_model = Sequential()
Facet_model.add(BatchNormalization(input_shape=(train_set_visual.shape[1],), name = 'facet_layer_0'))
Facet_model.add(Dropout(0.2, name = 'facet_layer_1'))
Facet_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'facet_layer_2', trainable=end_to_end))
Facet_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'facet_layer_3', trainable=end_to_end))
Facet_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'facet_layer_4', trainable=end_to_end))
Facet_model.summary()

text_model = Sequential()
text_model.add(LSTM(128, input_shape=(max_len, 300), name = 'text_layer_0', trainable=end_to_end))
text_model.add(Dense(64, name = 'text_layer_2', W_regularizer=l2(0.0), trainable=end_to_end))
text_model.summary()

bias_model=Sequential()
bias_model.add(Reshape((1,),input_shape=(1,)))
biased_Covarep = Merge([bias_model, Covarep_model], mode='concat')
biased_Facet = Merge([bias_model, Facet_model], mode='concat')
biased_text= Merge([bias_model, text_model], mode='concat')
Covarep_biased_model, Facet_biased_model, text_biased_model = Sequential(), Sequential(), Sequential()

Covarep_biased_model.add(biased_Covarep)
Covarep_biased_model.add(Reshape((1, 32 + 1)))
Facet_biased_model.add(biased_Facet)
Facet_biased_model.add(Reshape((1, 32 + 1)))
text_biased_model.add(biased_text)
text_biased_model.add(Reshape((1, 64 + 1)))
dot_layer1 = Merge([Covarep_biased_model, Facet_biased_model], mode='dot', dot_axes=1, name='dot_layer_1') 
dot_layer1_reshape = Reshape((1, (32 + 1) * (32 + 1)), name='5')
fusion_model_tmp = Sequential()
fusion_model_tmp.add(dot_layer1)
fusion_model_tmp.add(dot_layer1_reshape)
dot_layer2=Merge([fusion_model_tmp,text_biased_model], mode='dot', dot_axes=1, name='dot_layer_2')
fusion_model = Sequential()
fusion_model.add(dot_layer2)
fusion_model.add(Reshape(((32 + 1) * (32 + 1) * (64 + 1),)))
fusion_model.add(Dropout(args.dropout))
fusion_model.add(Dense(args.units, activation='relu', W_regularizer=l2(args.l2), name = 'fusion_layer_1'))
fusion_model.add(Dense(args.units, activation='relu', W_regularizer=l2(args.l2), name = 'fusion_layer_2'))
fusion_model.add(Dense(1, activation='sigmoid',W_regularizer=l2(args.l2), name = 'fusion_layer_4'))

#fusion_model.load_weights(weights_folder_path + "fusion-pretrained-cv" + str(cv_id) + '.h5', by_name=True)
callbacks = [
    EarlyStopping(monitor='val_acc', patience=5, verbose=1),
    ModelCheckpoint("pig.hdfs5", monitor='val_loss', save_best_only=True, verbose=1),
]
sgd = SGD(lr=args.lr, decay=1e-6, momentum=args.momentum, nesterov=True)
adam = optimizers.Adamax(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #decay=0.999)
optimizer = {'sgd': sgd, 'adam':adam}
fusion_model.compile(loss=args.loss, optimizer=optimizer[args.optimizer], metrics=['accuracy'])


fusion_model.fit([np.ones(train_set_audio.shape[0]), train_set_audio, train_set_visual, train_set_text], y_train, 
                 validation_data=[[np.ones(valid_set_audio.shape[0]), valid_set_audio, valid_set_visual, valid_set_text], y_valid],
                 nb_epoch=100, batch_size=50,
                 callbacks=callbacks
)

#predictions = fusion_model.predict([np.ones(valid_set_audio.shape[0]), valid_set_audio, valid_set_visual, valid_set_text], verbose=0)
score, acc = fusion_model.evaluate([np.ones(test_set_audio.shape[0]), test_set_audio, test_set_visual, test_set_text], y_test)
print(acc)

# predictions = predictions.reshape((len(y_test),))
# y_test = y_test.reshape((len(y_test),))
# mae = np.mean(np.absolute(predictions-y_test))
# print("mae: ", mae)
# print("corr: ", round(np.corrcoef(predictions,y_test)[0][1],5))
# print('mult_acc: ', round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5))
# true_label = (y_test >= 0)
# predicted_label = (predictions >= 0)
# print("Confusion Matrix :")
# print(confusion_matrix(true_label, predicted_label))
# print("Classification Report :")
# print(classification_report(true_label, predicted_label, digits=5))
# print("Accuracy ", accuracy_score(true_label, predicted_label))
# pdb.set_trace()

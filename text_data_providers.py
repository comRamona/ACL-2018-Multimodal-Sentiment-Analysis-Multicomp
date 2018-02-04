from data_providers import DataProvider
import os
import random
import numpy as np
import pandas as pd
from mmdata import MOSI
import logging

logger = logging.getLogger()
random.seed(1)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

class MOSITextProvider(DataProvider):
    """Data provider for MNIST handwritten digit images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, conv=True):
        """Create a new MNIST data provider object.
        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes =2 

        mosi = MOSI()
        embeddings = mosi.embeddings()
        sentiments = mosi.sentiments()
        if which_set == "train":
            input_ids = mosi.train()
        elif which_set == "valid":
            input_ids = mosi.valid()
        else:
            input_ids = mosi.test()
        maxlen = 15 # Each utterance will be truncated/padded to 15 words
        inputs = []
        targets = []

        logger.info("Preparing data...")
        for vid, vdata in embeddings['embeddings'].items(): # note that even Dataset with one feature will require explicit indexing of features
            for sid, sdata in vdata.items():
                if sdata == []:
                    continue
                example = []
                for i, time_step in enumerate(sdata):
                    # data is truncated for 15 words
                    if i == 15:
                        break
                    example.append(time_step[2]) # here first 2 dims (timestamps) will not be used

                for i in range(maxlen - len(sdata)):
                    example.append(np.zeros(sdata[0][2].shape)) # padding each example to maxlen
                example = np.asarray(example)
                label = 1 if sentiments[vid][sid] >= 0 else 0 # binarize the labels

                # here we just use everything except training set as the test set
                if vid in input_ids:
                    inputs.append(example)
                    targets.append(label)

        # Prepare the final inputs as numpy arrays
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        if conv:
            inputs = np.expand_dims(inputs, axis=3)
        # pass the loaded data to the parent class __init__
        super(MOSITextProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(MOSITextProvider, self).next()
        return inputs_batch, targets_batch

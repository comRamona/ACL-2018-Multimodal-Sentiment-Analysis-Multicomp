
import numpy as np
from mmdata import MOSI

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

def get_data(max_len_audio=20, max_len_text=15, max_len_visual=20):
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
    for vid in test_ids:
        for sid in embeddings['embeddings'][vid].keys():
            if embeddings['embeddings'][vid][sid] and facet['facet'][vid][sid] and covarep['covarep'][vid][sid]:
                test_set_ids.append((vid, sid))

    # partition the training, valid and tesembeddingsall sequences will be padded/truncated to 15 steps
    # data will have shape (dataset_size, max_len, feature_dim)
    max_len = max_len_audio
    train_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if covarep['covarep'][vid][sid]], axis=0)
    valid_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if covarep['covarep'][vid][sid]], axis=0)
    test_set_audio = np.stack([pad(covarep['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if covarep['covarep'][vid][sid]], axis=0)

    max_len = max_len_visual
    train_set_visual = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
    valid_set_visual = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
    test_set_visual = np.stack([pad(facet['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

    max_len = max_len_text
    train_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
    valid_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
    test_set_text = np.stack([pad(embeddings['embeddings'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)
    # binarize the sentiment scores for binary classification task
    y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
    y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0
    y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

    # train_set_audio = train_set_audio[:,:,1:35]
    # valid_set_audio = valid_set_audio[:,:,1:35]
    # test_set_audio = test_set_audio[:,:,1:35]

    visual_max = np.max(np.max(np.abs(train_set_visual), axis=0), axis=0)
    visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize this dimension
    train_set_visual = train_set_visual / visual_max
    valid_set_visual = valid_set_visual / visual_max
    test_set_visual = test_set_visual / visual_max

    train_set_visual[train_set_visual != train_set_visual] = 0
    valid_set_visual[valid_set_visual != valid_set_visual] = 0
    test_set_visual[test_set_visual != test_set_visual] = 0

    audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
    audio_max[audio_max==0] = 1
    train_set_audio = train_set_audio / audio_max
    valid_set_audio = valid_set_audio / audio_max
    test_set_audio = test_set_audio / audio_max

    train_set_audio[train_set_audio != train_set_audio] = 0
    valid_set_audio[valid_set_audio != valid_set_audio] = 0
    test_set_audio[test_set_audio != test_set_audio] = 0
    return train_set_audio, valid_set_audio, test_set_audio, train_set_text, valid_set_text, test_set_text, train_set_visual, valid_set_visual, test_set_visual, y_train, y_valid, y_test

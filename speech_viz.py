# Correction Matrix Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import numpy as np
from mmdata import Dataloader, Dataset
import scipy.io as sio
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from ggplot import aes,geom_point,ggtitle




labelsfile = "covarep_labels.txt"
f_limit = 36


def norm(feat, data):
    data = np.array([feature[2] for feature in data])
    data = data[:,:f_limit]
    n_rows = data.shape[0]
    dim = data.shape[1]
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    var = np.var(data,axis=0)
    if feat == "mean":
        return mean
    if feat == "var":
        return var
    if feat == "std":
        return std
    res = np.concatenate((mean, std, var),axis=0)
    return res



def load_data(f):
    input = open(labelsfile, "rb")
    label_names = input.read().split("\n")[:36]
#    print("\n".join(label_names))

    mosi = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSI')
    covarep = mosi.covarep()
    sentiments = mosi.sentiments() # sentiment labels, real-valued. for this tutorial we'll binarize them
    train_ids = mosi.train() # set of video ids in the training set
    valid_ids = mosi.valid() # set of video ids in the valid set
    test_ids = mosi.test() # set of video ids in the test set

    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
        for sid in covarep['covarep'][vid].keys():
            train_set_ids.append((vid, sid))

    train_set_audio = np.array([norm(f, covarep['covarep'][vid][sid]) for (vid, sid) in train_set_ids if covarep['covarep'][vid][sid]])
    data = pandas.DataFrame(data=train_set_audio)
    return label_names, data

def make_heat(feat, label_names, data):
    correlations = data.corr()
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(label_names),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(label_names, rotation='vertical', fontsize=6)
    ax.set_yticklabels(label_names, fontsize=8)
    plt.tight_layout()
    plt.suptitle("Speech Features Correlation: "+feat, fontsize=16)
    plt.savefig("speech_heat_"+feat+".pdf")


def tsne_vizualization(df, labels):
    rndperm = np.random.permutation(df.shape[0])
    n_sne = 7000
    feat_cols = [ 'pixel'+str(i) for i in range(len(labels)) ]
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)
    print("done fitting transform, plotting now")
    df_tsne = df.loc[rndperm[:n_sne],:].copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]    
    chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by digit")
    ggsave(plot = chart, filename = "TSNE_1_2_speech")
    
F = ["mean", "var", "std"]
for f in F[:1]:
    label_names, data = load_data(f)
#    make_heat(f, label_names, data)
    tsne_vizualization(data,label_names)

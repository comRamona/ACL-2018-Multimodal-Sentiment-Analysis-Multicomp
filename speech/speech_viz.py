# Correction Matrix Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import numpy as np
from mmdata import Dataloader, Dataset
import scipy.io as sio
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



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


def multiclass(data):
    new_data = []
    for item in data:
        if item <= -1.8:
            new_data.append(0)
        elif item <= -0.6:
            new_data.append(1)
        elif item <= 0.6:
            new_data.append(2)
        elif item <= 1.8:
            new_data.append(3)
        elif item <= 3.0:
            new_data.append(4)
    return new_data


def load_data(f):
    input = open(labelsfile, "rb")
    label_names = input.read().split("\n")[:36]
#    print("\n".join(label_names))

    mosi = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSI')
    covarep = mosi.covarep()
    sentiments = mosi.sentiments() # sentiment labels, real-valued. for this tutorial we'll binarize them
    train_ids = mosi.train() # set of video ids in the training set

    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
        for sid in covarep['covarep'][vid].keys():
            train_set_ids.append((vid, sid))

    train_set_audio = np.array([norm(f, covarep['covarep'][vid][sid]) for (vid, sid) in train_set_ids if covarep['covarep'][vid][sid]])
    train_set_audio[train_set_audio != train_set_audio] = 0
#    y_data = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
#    y_train_mc = multiclass(np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]))
    y_train_bin = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
# normalize covarep and facet features, remove possible NaN values
#    audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
#    train_set_audio = train_set_audio / audio_max
    x_data = pandas.DataFrame(data=train_set_audio,columns=label_names)
    return label_names, train_set_audio, x_data, y_train_bin



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


    
def tsne_vizualization(data, targets, labels):
#    target_names = ['strg_neg', 'weak_neg', 'neutral', 'weak_pos', 'strg_pos']
#    names_dict = {0:'strg_neg', 1:'weak_neg', 2:'neutral', 3:'weak_pos', 4:'strg_pos'}
    target_names = {False:'negative', True:'positive'}
    rndperm = np.random.permutation(data.shape[0])
    n_sne = 7000
    tsne = TSNE(n_components=2,random_state=0)
    data = np.nan_to_num(data)
    tsne_results = tsne.fit_transform(data)
#    print(targets)
    print(tsne_results)
    print("done fitting transform, plotting now")
    target_ids = range(len(target_names))
#    colors = ['r', 'g', 'b', 'c', 'm']
    colors = ['r', 'c']
    
    plt.figure()
    for i in range(len(tsne_results)):
        x = tsne_results[i,0]
        y = tsne_results[i,1]
        target = targets[i]
        color = colors[target]
        name = target_names[target]
        print(x,y,target,color,name)
        plt.scatter(x,y,c=color, label=name)
    plt.title("tSNE Dimensions and Sentiment Classes")        
    plt.legend(colors, target_names.values())
    plt.savefig("tSNE_speech.pdf")
    
#    plt.figure()
#    for i, c, label in zip(target_ids, colors, target_names):
#        plt.scatter(tsne_results[targets == i, 0], tsne_results[targets == i, 1], c=c, label=label)

#    x_tsne = tsne_results[:,0]
#    y_tsne = tsne_results[:,1]
#    heatmap, xedges, yedges = np.histogram2d(x_tsne, y_tsne, bins=50)
#    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#    plt.clf()
#    plt.title("t-SNE dimensions")
#    plt.imshow(heatmap.T, extent=extent, origin='lower')
    

    
F = ["mean", "var", "std"]
for f in F[:1]:
    label_names, data, df, y_data = load_data(f)
#    make_heat(f, label_names, data)
    tsne_vizualization(data,y_data,label_names)

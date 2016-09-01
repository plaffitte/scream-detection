"""
Created on Wed Jul 13 12:15:23 2016
@author: piero
"""

import numpy as np
import cPickle
import gzip
import os
import matplotlib.pyplot as plt

path = '/home/piero/Documents/Speech_databases/test/rnn_yun'
train_data = cPickle.load(gzip.open(os.path.join(path,"data/train.pickle.gz"),"rb"))
mask = train_data[2]
lab = train_data[1]
pred = cPickle.load(gzip.open(os.path.join(path,"training_pred.pickle.gz"),"rb"))
print(np.shape(train_data))
n_epoch = 100
n_batch = len(train_data[0])
n_streams = 5
n_classes = 3
found = False
count = 0.0
n_predictions = np.shape(pred)[0] * np.shape(pred)[1] * np.shape(pred)[2]
count_neg = 0.0

for i in range(len(pred)):
#    i = np.random.randint(0, n_batch)
    neg_mask = 0
    print(i%70)
    mask_i = mask[i%70]
    neg_mask = np.sum((mask_i==-1).ravel())
    count_neg += neg_mask
    lab_i = lab[i%70]
    lab_i = (-lab_i).argsort()
    lab_i = lab_i[:,:,0]
    pred_i = pred[i]
    pred_i = (-pred_i).argsort()
    pred_i = pred_i[:,:,0]
#    find = (lab_i==2)
    find = (pred_i==1)
    count += np.sum(find.ravel())
    print(np.sum(find.ravel()))
#    if any(find):
#        found = True
    
print(count_neg)
print("Percentage of time RNN predicted class Conversation:", count / (n_batch * 5 * 500 - count_neg))
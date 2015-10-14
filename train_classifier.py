# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:34:38 2015

Classifier training

@author: pierre
"""
import os
from read_wave import read_wave
from mfcc import get_mfcc
from EM_for_HMM import emHMM_algorithm
from read_raw import read_raw

def train_classifier(dataset,n_classes,Fs=16000):
    datalist = os.listdir(dataset)
    n_data_set = len(datalist)

    for i in range(n_data_set) :
        filepath = dataset + '/' + datalist[i]
        if datalist[i].find('.wav')==1 :
            try:
                [x, Fs, n_channels, n_samples] = read_wave(filepath)
            except:
                print(e.msg)

        elif datalist[i].find('.raw') :
            try:
                x = read_raw(filepath)
                if len(x.shape)>1:
                    n_channels = x.shape[1]
                    n_samples = x.shape[0]
                else:
                    n_channels = 1
                    n_samples = len(x)
            except:
                print(e.msg)
        features = get_mfcc(x,Fs)
        dimension = features.shape[1]
        size = features.shape[0]

        emHMM_algorithm(features,dimension,2,size)

train_classifier('/home/pierre/Documents/Speech_databases/an4/wav/an4_clstk/fash',2)


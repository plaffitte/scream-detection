# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:21:19 2015

@author: piero
"""
import os
import gzip
import cPickle
import matplotlib.pyplot as plt

def classNumberChange(N_init, N_new, data):
    f_train = gzip.open(os.path.join(data,'train.pickle'), 'rb')        
    f_valid = gzip.open(os.path.join(data,'valid.pickle'), 'rb')        
    f_test = gzip.open(os.path.join(data,'test.pickle'), 'rb')
    train_set = cPickle.load(f_train)
    valid_set = cPickle.load(f_valid)
    test_set = cPickle.load(f_test)
    
    if N_new == 2:
        for i in range(len(train_set[1])):
            if train_set[1][i] == max(train_set[1]):
                train_set[1][i] = 1
            elif train_set[1][i] == min(train_set[1]):
                train_set[1][i] = 0
        for i in range(len(valid_set[1])):
            if valid_set[1][i] == max(valid_set[1]):
                valid_set[1][i] = 1        
            elif valid_set[1][i] == min(valid_set[1]):
                valid_set[1][i] = 0
        for i in range(len(test_set[1])):
            if test_set[1][i] == max(test_set[1]):
                test_set[1][i] = 1        
            elif test_set[1][i] == min(test_set[1]):
                test_set[1][i] = 0
    elif N_new == 3:
        for i in range(len(train_set[1])):
            if train_set[1][i] == max(train_set[1]):
                train_set[1][i] = 2
            elif train_set[1][i] == min(train_set[1]):
                train_set[1][i] = 0
            else:
                train_set[1][i] = 1
        for i in range(len(valid_set[1])):
            if valid_set[1][i] == max(valid_set[1]):
                valid_set[1][i] = 2
            elif valid_set[1][i] == min(valid_set[1]):
                valid_set[1][i] = 0
            else:
                valid_set[1][i] = 1
        for i in range(len(test_set[1])):
            if test_set[1][i] == max(test_set[1]):
                test_set[1][i] = 2
            elif test_set[1][i] == min(test_set[1]):
                test_set[1][i] = 0
            else:
                test_set[1][i] = 1
 

#data = "/home/piero/Documents/Speech_databases/test/pickle_data"
#f_train = gzip.open(os.path.join(data,'train.pickle'), 'rb')        
#f_valid = gzip.open(os.path.join(data,'valid.pickle'), 'rb')        
#f_test = gzip.open(os.path.join(data,'test.pickle'), 'rb')
#train_set = cPickle.load(f_train)
#valid_set = cPickle.load(f_valid)
#test_set = cPickle.load(f_test)
#
#plt.plot(train_set[1])
#plt.show()
#
#classNumberChange(5,2,data)
#
#plt.plot(train_set[1])
#plt.show()
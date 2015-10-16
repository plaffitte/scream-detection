# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:29:22 2015

@author: piero
"""
import os
import cPickle
import gzip
import matplotlib.pyplot as plt

plt.ion()
#path = sys.argv[1]
path = '/home/piero/Documents/Experiments/dim_13/Spectral Coeff/Not Standardized'
os.chdir(path)
test_data, test_labels = cPickle.load(gzip.open('test.pickle.gz', 'rb'))
pred_data = cPickle.load(gzip.open('dnn.classify.pickle.gz','rb'))

shouting_data1=test_data[test_labels==1,35]
shouting_data2=test_data[test_labels==1,36]
noise_data1=test_data[test_labels==0,35]
noise_data2=test_data[test_labels==0,36]

plt.plot((noise_data1),(noise_data2),'r.')
plt.plot((shouting_data1),(shouting_data2),'b.')

print test_data.size

plt.figure()
plt.imshow(test_data[800:1000,:-1].T)

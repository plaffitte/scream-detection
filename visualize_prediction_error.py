# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:27:13 2015

@author: piero
"""
import os
import cPickle
import gzip
import matplotlib.pyplot as plt
import numpy as np

# path = sys.argv[1]
path = '/home/piero/Documents/Experiments/Real_Test/MFCC/Scream vs Noise/3*512'
os.chdir(path)
test_data, test_labels = cPickle.load(gzip.open(os.path.join(path, 'data/test.pickle.gz'), 'rb')) # cPickle.load(gzip.open('data/test.pickle.gz', 'rb'))
pred_data = cPickle.load(gzip.open('dnn.classify.pickle.gz', 'rb'))
prediction = (-pred_data).argsort()
predictionfinal = prediction[:, 0]
error_count = 0.0
error_vector = np.zeros(len(test_data))
noise_error = 0.0
shouting_error = 0.0
for i in xrange(len(test_data)):
    if predictionfinal[i] != test_labels[i]:
        error_count += 1
        if test_labels[i] == 0:
            noise_error += 1
        else:
            shouting_error += 1
    error_vector[i] = test_labels[i] - predictionfinal[i]
# y_axis = np.linspace(0,len(test_data),len(test_data))
# plt.plot(test_labels,'b')
plt.plot(error_vector, 'r')
plt.show()
error_rate = error_count/len(test_data)
print('length test data:', len(test_data))
print('length pred data:', len(pred_data))
print('total error rate: ', error_rate)
print('number of noise wrongly classified', noise_error)
print('number of scream wrongly classified', shouting_error)

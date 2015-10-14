# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:55:59 2015

@author: pierre
"""
# This script generates data and trains an SVM on it
# It then attempts to classify the data
import numpy as np
from sklearn import svm
from data_simulation import create_data

dimension = 1
number_of_components = 2
size = 1000


def main():

    if dimension == 1:
        x = np.zeros((1, number_of_components * size), dtype=float)
        mu = np.zeros((number_of_components, 1), dtype=float)
        sigma = np.zeros((number_of_components, 1, 1), dtype=float)

        for i in range(number_of_components):
            x[0, i * size:(i + 1) * size], mu[i, 0], sigma[i, 0,
                                                           0] = create_data(dimension, size, i)
    else:
        x = np.zeros((dimension, number_of_components * size), dtype=float)
        mu = np.zeros((number_of_components, dimension), dtype=float)
        sigma = np.zeros(
            (number_of_components, dimension, dimension), dtype=float)

        for i in range(number_of_components):
            x[:, i], mu[i, :], sigma[i,:,:] = create_data(dimension, size, i)

    model = svm.SVC()
    model.kernel = 'sigmoid'

    model.fit(x)

if __name__ == "__main__":
    main()

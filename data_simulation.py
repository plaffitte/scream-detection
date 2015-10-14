# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:54:09 2015

@author: pierre
"""
import numpy as np

def create_data(dimension=1, size=1000, i=0):

    if dimension == 1:
        if i == 0 :
            mu = 10
            sigma = 0.5
            x = mu + np.sqrt(sigma) * np.random.randn(1,size)
        elif i == 1:
            mu = 0.0
            sigma = 0.7
            x = mu + np.sqrt(sigma) * np.random.randn(1,size)
        elif i == 2:
            mu = 5
            sigma = 0.8
            x = mu + np.sqrt(sigma) * np.random.randn(1,size)
    elif dimension == 2:
        if i == 0 :
            mu = [-3,0.5]
            sigma = np.array([[0.8, 0.0],[0.0, 0.9]])
            x = mu + (1/np.sqrt(np.linalg.det(sigma))) * np.random.randn(size,dimension)
        elif i == 1:
            mu = [-5,4]
            sigma = np.array([[0.6,0.0],[0.0,0.95]])
            x = mu + (1/np.sqrt(np.linalg.det(sigma))) * np.random.randn(size,dimension)
        elif i == 2:
            mu = [2,9]
            sigma = np.array([[0.7,0.0],[0.0,0.8]])
            x = mu + (1/np.sqrt(np.linalg.det(sigma))) * np.random.randn(size,dimension)
#    if dimension == 1 :
#        mu = np.random.randint(1,20)
#        sigma = 10*np.random.random_sample(1)
#        x = np.random.normal(mu,sigma,(1,size))
#    else :
#        mu = np.floor(10*np.random.random(dimension))
#        sigma = np.zeros((dimension,dimension))
#        for j in  range(dimension):
#            sigma[j,j] = 10*np.random.random_sample(1)
#            mu[j] = np.random.randint(1,10)
#            x = np.random.multivariate_normal(mu,sigma,size)

    return x, mu, sigma
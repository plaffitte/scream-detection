# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 18:51:56 2014

@author: pierre
"""
## This script simulates data from several gaussian distributions with different parameters
## and learns the parameters with an HMM model via the EM algorithm.
## Each HMM state is represented by a single Gaussian distribution
## It also calculates the sequence of states corresponding to the data via a Viterbi algorithm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from EM_for_HMM import emHMM_algorithm
from viterbi import viterbi
from sklearn import hmm
import math
from data_simulation import create_data
import cPickle
import gzip
import dnn
from theano.tensor.shared_randomstreams import RandomStreams
# Variable definitions
dimension=1
number_of_components=2
size=1000

def main():

    if dimension==1 :
#        gmm = np.zeros(number_of_components*size)
#        mu = np.zeros(number_of_components)
#        sigma = np.zeros(number_of_components)
#        for i in range(number_of_components) :
#            gmm[i*size:(i+1)*size], mu[i], sigma[i] = create_data(dimension,size,i)
        gmm = np.zeros((1,number_of_components*size),dtype=float)
        mu = np.zeros((number_of_components,1),dtype=float)
        sigma = np.zeros((number_of_components,1,1),dtype=float)
        matrix = np.zeros((number_of_components,number_of_components),dtype=float)

#        for i in range(number_of_components):
#            x, mu[i,0], sigma[i,0,0] = create_data(dimension,size,i)
    else:
        gmm = np.zeros((dimension,number_of_components*size),dtype=float)
        mu = np.zeros((number_of_components,dimension),dtype=float)
        sigma = np.zeros((number_of_components,dimension,dimension),dtype=float)
        matrix = np.zeros((number_of_components,number_of_components),dtype=float)

#        for i in range(number_of_components):
#            x, mu[i,:], sigma[i,:,:] = create_data(dimension,size,i)

    weights = np.array([0.6, 0.4])
    matrix = np.array([[0.7, 0.3], [0.1, 0.9]])
    model = hmm.GaussianHMM(2, "full", weights, matrix)
    model.means_ = mu
    model.covars_ = sigma
    gmm, Z = model.sample(number_of_components*size)

#    else :
#        gmm = np.zeros((dimension,number_of_components*size))
#        mu = np.zeros((number_of_components,dimension))
#        sigma = np.zeros((number_of_components,dimension,dimension))
#        for i in range(number_of_components) :
#            gmm[:,i*size:(i+1)*size], mu[i,:], sigma[i,:,:] = create_data(dimension,size,i)

    means, variances, pi, a = emHMM_algorithm(gmm,dimension,number_of_components,number_of_components*size)

#    num_bins = 50
#    n, bins, patches = plt.hist(gmm, num_bins, normed=1, facecolor='green', alpha=0.5)
#    # add a 'best fit' line
#    for i in range(number_of_components) :
#        y = mlab.normpdf(bins, means[i], variances[i])
#        plt.plot(bins, y, 'r--')
#        plt.xlabel('Values')
#        plt.ylabel('Probability')
#        plt.title('Data Histogram vs predicted distribution')
#
#    # Tweak spacing to prevent clipping of ylabel
#    plt.subplots_adjust(left=0.15)
#    plt.show()

    b = np.zeros((number_of_components,number_of_components*size))

    #Evaluate posterior
    if dimension==1:
        for i in range(number_of_components) :
        # Calculate the probability of seeing the observation given each state
            pdf = pi[i]*mlab.normpdf(gmm, means[i], variances[i,0])
            b[i,:] = pdf[:,0]

    else:
        centered_data = np.zeros((number_of_components,number_of_components*size,dimension))
        den = np.zeros((number_of_components,number_of_components*size))
        num = np.zeros((number_of_components,number_of_components*size))
        for i in range(number_of_components) :
        # Calculate the probability of seeing the observation given each state
            for n in range(number_of_components*size):
                centered_data[i, n, :] = gmm[n, :]-means[i, :]
                den[i,n] = np.sqrt((2*math.pi)**(dimension)*np.linalg.det(variances[i,:,:]))
                num[i,n] = np.exp((-1/2)*np.dot(np.dot(centered_data[i,n,:][np.newaxis],np.linalg.inv(variances[i,:,:])),centered_data[i,n,:][:,np.newaxis]))
                b[i,n] = num[i,n] / den[i,n]


    # Predict
    path, x, y = viterbi(size*number_of_components,a,b,pi)
    plt.figure();
    plt.plot(path[0,:],'ro')
    plt.plot(path[0,:],'r')
    plt.plot(Z,'g')
    plt.show()
    if dimension==1:
        print "initial means: ", mu[:,0], "\n", "initial variances: ", sigma[:,0,0], "\n", "initial weights: ", weights, "\n"
        print "means:", means, "\n" "sigmas:", variances, "\n", "weights:", pi, "\n"
        print "initial mixing mgmmatrix:", matrix, "\n"
        print "mixing matrix:", a, "\n"
    else:
        print "initial means: ", mu, "\n", "initial variances: ", sigma, "\n", "initial weights: ", weights, "\n"
        print "means:", means, "\n" "sigmas:", variances, "\n", "weights:", pi, "\n"
        print "initial mixing matrix:", matrix, "\n"
        print "mixing matrix:", a, "\n"

if __name__ == "__main__":
    main()
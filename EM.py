# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 18:51:56 2014

@author: pierre
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from EM_for_HMM import *

def main():
    
    #main function
    #def run_em(dimension=1, number_of_components=2, size=1000):
    dimension=1
    number_of_components=3
    size=10000
    if dimension==1 :
            gmm = np.zeros(number_of_components*size)
            mu = np.zeros(number_of_components)
            sigma = np.zeros(number_of_components)
            for i in range(number_of_components) :
                gmm[i*size:(i+1)*size], mu[i], sigma[i] = create_data(dimension,size,i)
    else :
            gmm = np.zeros((dimension,number_of_components*size))
            mu = np.zeros((number_of_components,dimension))
            sigma = np.zeros((number_of_components,dimension,dimension))
            for i in range(number_of_components) :
                gmm[:,i*size:(i+1)*size], mu[i,:], sigma[i,:,:] = create_data(dimension,size,i)
        
    means, variances, weights = emHMM_algorithm(gmm,dimension,number_of_components,number_of_components*size)
    
    num_bins = 50
    n, bins, patches = plt.hist(gmm, num_bins, normed=1, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    for i in range(number_of_components) :
        y = mlab.normpdf(bins, means[i], variances[i])
        plt.plot(bins, y, 'r--')
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title('Data Histogram vs predicted distribution')
    
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()
    
    print "initial means are: ", mu, "\n", "initial variances are: ", sigma, "\n"
    print "means:", means, "\n" "sigmas:", variances, "\n", "weights:", weights, "\n"

    
def create_data(dimension=1, size=1000, i=0):
    
    if dimension == 1 :
        mu = np.random.randint(1,10)
        sigma = 10*np.random.random_sample(1)
        x = np.random.normal(mu,sigma,size)
    else :
        mu = np.floor(10*np.random.random(dimension))
        sigma = np.zeros((dimension,dimension))                 
        for j in  range(dimension):
            sigma[j,j] = 10*np.random.random_sample(1)
            mu[j] = np.random.randint(1,10)
            x = np.random.multivariate_normal(mu,sigma,size)
        x = np.sqrt(sigma)*np.random.randn(size,dimension) + mu

#    if i==0 :
#        mu=0
#        sigma=np.array([[1]])
#    elif i==1 :
#        mu=5
#        sigma=np.array([[0.5]])
#    elif i==2 :
#        mu=-3
#        sigma=np.array([[1]])
    
    x = np.sqrt(sigma)*np.random.randn(size,dimension) + mu

    return x.T, mu, sigma
    


def em_gmm(data,dimension,number_of_components,size):
    stop_criteria = 0
    ML = 0
    it = 0
    prob_data_for_all_latent = np.zeros(size)
    if dimension==1 :
#        calculate mean
#        kmeans = KMeans(number_of_components,100,10)
#        kmeans.fit(data)
        data_mean = np.mean(data)
        data_var = np.var(data)
        means = np.random.normal(data_mean,data_var/number_of_components,number_of_components)
        sigmas = np.random.normal(data_var/number_of_components,data_var/number_of_components,number_of_components)
        pi = 1/number_of_components*np.ones(number_of_components)
        prob_data_for_all_latent = np.zeros(size)
        prob_data_given_latent = np.zeros((number_of_components,size))
        gamma = np.zeros((number_of_components,size))
        N = np.zeros(number_of_components)
        centered_data = np.zeros((number_of_components,size))
#        while stop_criteria!=1:
        for cnt in range(400) :
            it += 1
            for j in range(number_of_components) :
                prob_data_given_latent[j,:] = pi[j]/(math.sqrt(2*math.pi*sigmas[j]))*np.exp(-(data[:]-means[j])**2/(2*sigmas[j]))
            prob_data_for_all_latent = np.sum(prob_data_given_latent,0)
            gamma = prob_data_given_latent/prob_data_for_all_latent
            N = np.sum(gamma,1)
            
            for k in range(number_of_components) :
                centered_data = data - means[k]
                sigmas[k] = (1/N[k])*np.sum((gamma[k,:]*centered_data**2),0)
            
            pi = N/size               
            ML_prev = ML
            ML = np.log(sum(prob_data_for_all_latent))
            ML_var = ML_prev - ML
#            if ML_var>0:
#                break
            if abs(ML_var)<=10**-10:
                stop_criteria = 1
            print("it : " + str(it) + " " + "ML = " + str(ML))
#            print("poids: " + str(pi))
#            print("moyennes" + str(means))
#            if ML_var==0:
#                stop_criteria = 1
    else :
#        data_mean = np.zeros(dimension)
        data_var = np.zeros((dimension,dimension))
        data_mean = np.mean(data,1)
        variance = np.var(data,1)
        sigmas = np.zeros((number_of_components,dimension,dimension))
        means = np.zeros((number_of_components,dimension))
        for ii in range(number_of_components) :
            for i in range(dimension) :
                data_var[i,i] = variance[i]
                sigmas[ii,i,i] = np.random.normal(data_var[i,i]/number_of_components,data_var[i,i]/number_of_components)
            means[ii,:] = np.random.multivariate_normal(data_mean,data_var/number_of_components,1)
        pi = 0.5*np.ones(number_of_components)
        prob_data_for_all_latent = np.zeros(size)
        prob_data_given_latent = np.zeros((number_of_components,size))
        gamma = np.zeros((number_of_components,size))
        N = np.zeros(number_of_components)
        centered_data = np.zeros((number_of_components,dimension,size))
        centered_samples = np.zeros((number_of_components,dimension,size))
        while stop_criteria!=1:
            it += 1
            for j in range(number_of_components) :
                for n in range(size) :
                    centered_data[j,:,n] = data[:,n]-means[j,:]  
                    if np.linalg.det(sigmas[j,:,:])!=0:
                        exponential = np.exp(-1/2*np.dot(centered_data[j,:,n].T,np.dot(np.linalg.inv(sigmas[j,:,:]),centered_data[j,:,n])))
                        prob_data_given_latent[j,n] = pi[j]/((2*math.pi)**(1/dimension)*abs(np.linalg.det(sigmas[j,:,:]))**(1/2))*exponential
                    else:
                        augmented_matrix = sigmas[j,:,:] + 10**-9  
                        exponential = np.exp(-1/2*np.dot(centered_data[j,:,n].T,np.dot(np.linalg.inv(augmented_matrix),centered_data[j,:,n])))
                        prob_data_given_latent[j,n] = pi[j]/((2*math.pi)**(1/dimension)*abs(np.linalg.det(augmented_matrix))**(1/2))*exponential
                    prob_data_for_all_latent = np.sum(prob_data_given_latent,0)
            gamma = prob_data_given_latent/prob_data_for_all_latent
            N = np.sum(gamma,1)
            pi = N/size
            ML_prev = ML
            ML = np.log(sum(prob_data_for_all_latent))
            ML_var = ML_prev - ML
            if ML_var>0:
                exit
            if abs(ML_var)<=10**-9:
                stop_criteria = 1
            print("it : " + str(it) + " " + "ML = " + str(ML))

            for ll in range(number_of_components) :
                prod = np.zeros((number_of_components,size))
                for l in range(size) :
                    prod[:,l] = gamma[ll,l]*data[:,l]
                means[ll,:] = (np.sum(prod,1))/N
            for k in range(number_of_components) :
                product = np.zeros((size,dimension,dimension))
                sum_inside_sigma = np.zeros((number_of_components,dimension,dimension))
                interm_tab = np.zeros((1,dimension))
                for kk in range(size) :
                    centered_samples[k,:,kk] = data[:,kk] - means[k,:]
                    interm_tab[0,:] = centered_samples[k,:,kk]
                    square_vector_multiplication = interm_tab.T*interm_tab
                    product[kk,:,:] = gamma[k,kk]*square_vector_multiplication
                sum_inside_sigma[k,:,:] = np.sum(product,0)
                sigmas[k,:,:] = (1/N[k])*sum_inside_sigma[k,:,:]
    return means, sigmas, pi

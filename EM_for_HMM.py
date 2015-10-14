# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 19:12:33 2015

@author: pierre
"""
from sklearn.cluster import KMeans
import numpy as np
from FB_algo import FB_algo
import math

def emHMM_algorithm(data,dimension,number_of_components,size):
    stop_criteria = 0
    Like0 = 0
    it = 0
    if dimension==1 :
        means = np.zeros(number_of_components)
        #calculate mean
        kmeans = KMeans(number_of_components,'k-means++',10,100)
        kmeans.fit(data)
        for i in range(number_of_components) :
            means[i] = kmeans.cluster_centers_[i]
        sigmas = np.ones(number_of_components)
        pi = 1/float(number_of_components)*np.ones(number_of_components)
        psi = np.zeros((number_of_components,number_of_components,size))
        a = 0.5*np.ones((number_of_components,number_of_components)) # transition matrix, columns represent the initial states, lines represent the final states
        b = np.zeros((number_of_components,size))
        gamma = np.zeros((number_of_components,size-1))
        centered_data = np.zeros((number_of_components,size))
        while stop_criteria!=1:
            it += 1
            for i in range(number_of_components) :
                b[i,:] = pi[i]/(math.sqrt(2*math.pi*sigmas[i]))*np.exp(-(data[:,0]-means[i])**2/(2*sigmas[i]))

            alpha, beta = FB_algo(size,a,pi,b) # run the Forward-Backward procedure

            # Calculate Psi and Gamma
            for t in range(size-1) :
                alpha_t = alpha[:,t][:,np.newaxis]
                b_t = b[:,t+1][np.newaxis]
                beta_t = beta[:,t+1][np.newaxis]
                x = alpha_t*a*b_t*beta_t
                psi[:,:,t] = x/np.sum(np.sum(x,0))

            gamma = np.sum(psi,1)

            # Update means and covariances
            means = (np.sum((gamma[:,:]*data[:,0]),1))/np.sum(gamma[:,:],1)

            for k in range(number_of_components) :
                centered_data = data[:,0] - means[k]
                sigmas[k] = np.sum((gamma[k,:]*centered_data[:]**2),0)/np.sum(gamma[k,:],0)

            # Update transition matrix and weights
            a = np.sum(psi,2)/(np.sum(gamma[:,:],1)[:,np.newaxis])
            pi = np.sum(gamma[:,:],1)/size

            # Calculate Likelihood
            ML_prev = Like0
            Like0 = np.sum(np.log(np.sum(b,0)))
            ML_var = ML_prev - Like0
#            if ML_var>0:
#                break
            if abs(ML_var)<=10**-4:
                stop_criteria = 1
            print("it : " + str(it) + " " + "ML = " + str(Like0))
            if ML_var==0:
                stop_criteria = 1

    else:
        means = np.zeros((number_of_components, dimension))
        #calculate mean
        kmeans = KMeans(number_of_components, 'k-means++', 10, 100)
        kmeans.fit(data)
        for i in range(number_of_components):
            means[i, :] = kmeans.cluster_centers_[i]
        sigmas = np.tile(np.identity(dimension), (number_of_components, 1, 1))
        pi = 1/float(number_of_components)*np.ones(number_of_components)
        psi = np.zeros((number_of_components, number_of_components, size))
        a = 0.5*np.ones((number_of_components, number_of_components))
        #transition matrix
        #columns represent the initial states, lines represent the final states
        gamma = np.zeros((number_of_components, size))
        centered_data = np.zeros((number_of_components, size, dimension))
        b = np.zeros((number_of_components, size))
        while stop_criteria != 1:
            it += 1
            for i in range(number_of_components):
                for n in range(size):
                    centered_data[i, n, :] = data[n, :]-means[i, :]
                    if np.linalg.det(sigmas[i,:,:])!=0:
                        exponential = np.exp(-1/2*np.dot(centered_data[i,n,:].T,np.dot(np.linalg.inv(sigmas[i,:,:]),centered_data[i,n,:])))
                        b[i,n] = pi[i]/((2*math.pi)**(1/dimension)*abs(np.linalg.det(sigmas[i,:,:]))**(1/2))*exponential
                    else:
                        augmented_matrix = sigmas[i,:,:] + 10**-1
                        exponential = np.exp(-1/2*np.dot(centered_data[i,n,:].T,np.dot(np.linalg.inv(augmented_matrix),centered_data[i,n,:])))
                        b[i,n] = pi[i]/((2*math.pi)**(1/dimension)*abs(np.linalg.det(augmented_matrix))**(1/2))*exponential

            alpha, beta = FB_algo(size,a,pi,b) # run the Forward-Backward procedure

            # Calculate Psi and Gamma
            for t in range(size-1) :
                alpha_t = alpha[:,t][:,np.newaxis]
                b_t = b[:,t+1][np.newaxis]
                beta_t = beta[:,t+1][np.newaxis]
                x = alpha_t*a*b_t*beta_t
                psi[:,:,t] = x/np.sum(np.sum(x,0))

            gamma = np.sum(psi,1)
            N = np.sum(gamma,1)

#            # Update means and covariances
#            for ii in range(number_of_components):
#                means[ii,:] = np.sum((gamma[ii,:][:,np.newaxis]*data),0)/np.sum(gamma[ii,:])
#
#            for k in range(number_of_components) :
#                centered_data[k,:,:] = data - means[k,:][np.newaxis]
#                sigmas[k,:,:] = np.sum((gamma[k,:][:,np.newaxis]*centered_data[k,:,:]**2),0)/np.sum(gamma[k,:],0)

            for ii in range(number_of_components) :
                means[ii,:] = np.sum((gamma[ii,:][:,np.newaxis]*data),0)/np.sum(gamma[ii,:])

            for k in range(number_of_components) :
                product = np.zeros((size,dimension,dimension))
                sum_inside_sigma = np.zeros((number_of_components,dimension,dimension))
                interm_tab = np.zeros((1,dimension))
                for kk in range(size) :
                    centered_data[k,kk,:] = data[kk,:] - means[k,:]
                    interm_tab[0,:] = centered_data[k,kk,:]
                    square_vector_multiplication = interm_tab.T*interm_tab
                    product[kk,:,:] = gamma[k,kk]*square_vector_multiplication
                sum_inside_sigma[k,:,:] = np.sum(product,0)
                sigmas[k,:,:] = (1/N[k])*sum_inside_sigma[k,:,:]

            # Update transition matrix and weights
            a = np.sum(psi,2)/(np.sum(gamma[:,:],1)[:,np.newaxis])
            pi = np.sum(gamma[:,:],1)/size

            # Calculate Likelihood
            ML_prev = Like0
            Like0 = np.sum(np.log(np.sum(b,0)))
            ML_var = ML_prev - Like0
#            if ML_var>0:
#                break
            if abs(ML_var)<=10**-2:
                stop_criteria = 1
            print("it : " + str(it) + " " + "ML = " + str(Like0))
            if ML_var==0:
                stop_criteria = 1

    return means, sigmas, pi, a

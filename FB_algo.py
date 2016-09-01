# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:53:30 2015
@author: pierre
"""
import numpy as np

def FB_algo(L, a, pi, b):
    
    N = pi.size #number of classes
    alpha = np.zeros((N,L))
    beta = np.zeros((N,L))
    scale_coef = np.ones(L-1) 
    alpha_t = np.zeros((N,1))
    beta_t = np.zeros((N,1))
        
    # Init
    alpha[:,0] = (pi*b[:,0])/(np.sum(pi*b[:,0],0))
    beta[:,L-1] = np.ones(N);
    
    # alpha update
    for t in range(0,L-1,1) :
        alpha_t = alpha[:,t][:,np.newaxis]
        scale_coef[t] = np.sum((np.sum(alpha_t*a,0))*b[:,t+1])
        alpha[:,t+1] = (np.sum(alpha_t*a,0)*b[:,t+1])/scale_coef[t]
        
    for t in range(L-2,-1,-1) :
        beta_t = beta[:,t+1][:,np.newaxis]
        b_t = b[:,t+1][np.newaxis]
        beta[:,t] = np.sum(a*b_t*beta_t,0)/scale_coef[t]
        
    return alpha, beta
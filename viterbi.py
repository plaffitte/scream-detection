# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 15:31:48 2015

@author: pierre
"""
import numpy as np
def viterbi(L, a, b, pi):
    
    N = pi.shape[0] #number of classes
    delta_t_1 = np.zeros((N,1),dtype=float)
    delta = np.zeros((N,L),dtype=float)
    phi = np.zeros((N,L))
    path = np.zeros((1,L),dtype=int)
        
    # Init
    delta[:,0] = np.log(pi[:]*b[:,0])
    
    # Delta update 
    for t in range(1,L,1) :
        delta_t_1 = delta[:,t-1][:,np.newaxis]
        log_int_var = delta_t_1 + np.log(a)
        b_vector = np.log(b[:,t][np.newaxis])
#        for k in range(N):
#            buffer_vec = int_var[0,k]
#            phi[k,t] = 0
#            for j in range(N):
#                if int_var[j,k] > buffer_vec:
#                    phi[k,t] = j
#                    buffer_vec = int_var[j,k]
        
        delta[:,t] = np.amax(log_int_var+b_vector,0)
        phi[:,t] = np.argmax(log_int_var+b_vector,0)
    
    delta_L = delta[:,L-1][:,np.newaxis]
    path[0,L-1] = np.argmax(delta_L)
    phi[0,L-1] = path[0,L-1]
    phi[1,L-1] = path[0,L-1]
#    for k in range(N):
#        buffer_vec = delta_L[0]
#        path[L-1] = 0
#        if delta_L[k] > buffer_vec:
#            path[L-1] = j
#            buffer_vec = int_var[j,k]
    for t in range(L-2,-1,-1):
        path[0,t] = phi[int(path[0,t+1]),t+1]
        
    return path, phi, delta
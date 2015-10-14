# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 12:05:23 2015
Operates audio data transformation, specifically, calculate mfcc coefficients

@author: pierre laffitte
"""

from features import mfcc

def get_mfcc(x,fs=48000,**kwargs):
               
    vector_mfcc = mfcc(x,fs,**kwargs)
    
    return vector_mfcc
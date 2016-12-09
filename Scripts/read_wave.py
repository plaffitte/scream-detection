# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 12:25:40 2015

Read audio file

@author: pierre
"""
import wave 

def read_wave(filepath):
    
    try:
        x = wave.open(filepath)
        nchannels = x.getnchannel()
        fs = x.getframerate()
        nsamples = x.getnframes()
        x = x.readframes(nsamples)
    except Exception as e:
        raise e
        
    return x, fs, nchannels, nsamples
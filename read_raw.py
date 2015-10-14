# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 16:37:30 2015

@author: pierre
"""

import numpy

def read_raw(filepath):
    data = numpy.memmap(filepath, dtype='h', mode='r')
    size = data.shape[0]
    x = numpy.zeros((size))
    for i in range(size):
        x[i] = data[i]
    return x
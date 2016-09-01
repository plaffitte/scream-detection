# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:35:23 2015

@author: piero
"""
import fileinput

def strreplace(str1,str2):
    filename = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/lab/1421675810481.lab'
    for line in fileinput.FileInput(filename,inplace=1):
        line = line.replace(str1,str2)
        print line,

strreplace('conversation','Conversation')

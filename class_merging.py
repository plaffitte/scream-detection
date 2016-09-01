# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:07:54 2015

@author: piero
"""
import os
import fileinput

def class_merging(label_folder,labels_to_merge):
    if type(labels_to_merge)!=dict:
        print('labels to merge must be provided in a dictionnary with pairs of labels to be merged.\n')
        print('Only 2 labels can be merged together\n')
    else:
        file_list = os.listdir(label_folder)        
        for i in range(len(file_list)):
            label_file = os.path.join(label_folder,file_list[i])
#            f = open(label_file,'r')
            for line in fileinput.FileInput(label_file,inplace=True):
                for label_1, label_2 in labels_to_merge.items():
#                        f.write(line.replace(label_2, label_1))
                        line = line.replace(label_1,label_2)
                        print line,

class_merging('/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/test_labels',{'Shout':'Screams'})
#class_merging('/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/test_labels',{'Conversation':'Other'})
#class_merging('/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/test_labels',{'BG_conversation':'Other'})

            
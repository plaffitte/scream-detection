#!/bin/bash
pythoncmd="nice -n19  /usr/bin/python -u"
pdnndir=/home/piero/Documents/Softwares/pdnn  # pointer to PDNN
theanodir=/usr/local/lib/python2.7/dist-packages/theano
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs

# D = Noisy dataset of Conversation and Shouting, with double labels (4 classes: noisy, not_noisy, conversation, shouting)
# C = Clean Dataset
# DNN 1 = Noisy (Conversation+Shouting) versus !Noisy (Conversation+Shouting)
# DNN 2 = Conversation vs Shout in Clean env.
# DNN 3 = Conversation vs Shout in Noisy env.

############# Create Dataset D
# Create dataset with occurrences from classes conversation and shouting
classes="{Conversation,Shouting}"
constraints="{Noisy,Not_noisy}"
# Create dataset from previously created dataset with labels 'noisy' and 'not_noisy'
$pythoncmd /home/piero/Documents/Scripts/format_noisy_dataset.py --classes $classes --constraints $constraints

############# Create Dataset C

############# Train DNN 1 with dataset D

############# Train DNN 2 with dataset C

############# Train DNN 3 with dataset D

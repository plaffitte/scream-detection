# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:27:28 2015

@author: pierre
"""

import cPickle
import gzip
import os
import sys
import time
from data_simulation import create_data
import numpy as np
import theano
import theano.tensor as T
import dnn_run
from theano.tensor.shared_randomstreams import RandomStreams
from models.dnn import DNN
from models.dropout_nnet import DNN_Dropout
from io_func.model_io import _nnet2file, _cfg2file, _file2nnet, log
from utils.utils import parse_arguments
from utils.learn_rates import _lrate2file, _file2lrate
from utils.network_config import NetworkConfig
from learning.sgd import train_sgd, validate_by_minibatch

arguments['train_data'] = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/wav/1421675810481.pickle'
arguments['valid_data'] = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/wav/1421675810481.pickle'
arguments['nnet_spec'] = '784:1024:1024:10'
arguments['param-output-file'] = '/home/piero/Documents/Deep_Learning_results/dnn_params' # dnn.param
arguments['cfg-output-file'] = '/home/piero/Documents/Deep_Learning_results/dnn_config'
arguments['layer-index'] = -1
arguments['wdir'] = '/home/piero/Documents/Deep_Learning_results/'

dnn_run.dnn_run(arguments)



## Extract features from test data
#arguments_feat_ext['data'] = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/wav/1421675810481.pickle'
#arguments_feat_ext['nnet-param'] = '/home/piero/Documents/Deep_Learning_results/dnn_params'
#arguments_feat_ext['nnet-cfg'] = dnn.cfg
#arguments_feat_ext['output-file'] = "dnn.classify.pickle.gz"
#run_Extract_Feats(arguments_feat_ext)
# Copyright 2014    Yajie Miao    Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from models.dnn import DNN
from models.dropout_nnet import DNN_Dropout
from io_func.model_io import _nnet2file, _cfg2file, _file2nnet
from utils.utils import parse_arguments
from utils.utils import format_results
from utils.learn_rates import _lrate2file, _file2lrate
from utils.network_config import NetworkConfig
from learning.sgd import train_sgd, validate_by_minibatch
from util_func import log, parse_classes

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['train_data', 'valid_data', 'nnet_spec', 'wdir']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']
    valid_data_spec = arguments['valid_data']
    nnet_spec = arguments['nnet_spec']
    wdir = arguments['wdir']
    multi_label = arguments['multi_label']
    if multi_label=="true": multi_label = True
    else:   multi_label = False

    # parse network configuration from arguments, and initialize data reading
    cfg = NetworkConfig(multi_label)
    cfg.parse_config_dnn(arguments, nnet_spec)
    cfg.init_data_reading(train_data_spec, valid_data_spec)

    # parse pre-training options
    # pre-training files and layer number (how many layers are set to the pre-training parameters)
    ptr_layer_number = 0; ptr_file = ''
    if arguments.has_key('ptr_file') and arguments.has_key('ptr_layer_number'):
        ptr_file = arguments['ptr_file']
        ptr_layer_number = int(arguments['ptr_layer_number'])

    # check working dir to see whether it's resuming training
    resume_training = False
    if os.path.exists(wdir + '/nnet.tmp') and os.path.exists(wdir + '/training_state.tmp'):
        resume_training = True
        cfg.lrate = _file2lrate(wdir + '/training_state.tmp')
        log('> ... found nnet.tmp and training_state.tmp, now resume training from epoch ' + str(cfg.lrate.epoch))

    numpy_rng = numpy.random.RandomState()
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... building the model')
    # setup model
    if cfg.do_dropout:
        dnn = DNN_Dropout(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)
    else:
        dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)

    # initialize model parameters
    # if not resuming training, initialized from the specified pre-training file
    # if resuming training, initialized from the tmp model file
    if (ptr_layer_number > 0) and (resume_training is False):
        _file2nnet(dnn.layers, set_layer_num = ptr_layer_number, filename = ptr_file)
    if resume_training:
        _file2nnet(dnn.layers, filename = wdir + '/nnet.tmp')

    # get the training, validation and testing function for the model
    log('> ... getting the finetuning functions')
    train_fn, valid_fn = dnn.build_finetune_functions(
                (cfg.train_x, cfg.train_y), (cfg.valid_x, cfg.valid_y),
                batch_size=cfg.batch_size)

    log('> ... finetuning the model')
    while (cfg.lrate.get_rate() != 0):
        # one epoch of sgd training
        train_error, pred, labels = train_sgd(train_fn, cfg)
        n_data_train = len(pred)
        log("-->> Epoch %d, \n " % cfg.lrate.epoch)
        log("- Training:\n")
        if multi_label:
            format_results(train_error, pred, labels, multi_label, cfg)
        else:
            format_results(train_error, pred, cfg.train_sets.label_vec, multi_label, cfg)
        # validation
        valid_error, pred2, labels = validate_by_minibatch(valid_fn, cfg)
        n_data_valid = len(pred2)
        log("- Validation:\n")
        if multi_label:
            format_results(valid_error, pred2, labels, multi_label, cfg)
        else:
            format_results(valid_error, pred2, cfg.valid_sets.label_vec, multi_label, cfg)
        cfg.lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))
        # output nnet parameters and lrate, for training resume
        if cfg.lrate.epoch % cfg.model_save_step == 0:
            _nnet2file(dnn.layers, filename=wdir + '/nnet.tmp')
            _lrate2file(cfg.lrate, wdir + '/training_state.tmp')

    # save the model and network configuration
    for i in range(len(cfg.hidden_layers_sizes)):
        if i==0:
            n_params = (cfg.n_ins + 1) * cfg.hidden_layers_sizes[i]
        else:
            n_params += (cfg.hidden_layers_sizes[i-1] + 1) * cfg.hidden_layers_sizes[i]
    n_params += cfg.n_outs * (cfg.hidden_layers_sizes[i-1] + 1)
    ratio = float(n_params) / float(n_data_train)
    log('-->> Ratio Parameters / Data : ' + str(ratio))
    if cfg.param_output_file != '':
        _nnet2file(dnn.layers, filename=cfg.param_output_file, input_factor = cfg.input_dropout_factor, factor = cfg.dropout_factor)
        log('> ... the final PDNN model parameter is ' + cfg.param_output_file)
    if cfg.cfg_output_file != '':
        _cfg2file(dnn.cfg, filename=cfg.cfg_output_file)
        log('> ... the final PDNN model config is ' + cfg.cfg_output_file)

    # output the model into Kaldi-compatible format
    if cfg.kaldi_output_file != '':
        dnn.write_model_to_kaldi(cfg.kaldi_output_file)
        log('> ... the final Kaldi model is ' + cfg.kaldi_output_file)

    # remove the tmp files (which have been generated from resuming training)
    os.remove(wdir + '/nnet.tmp')
    os.remove(wdir + '/training_state.tmp')

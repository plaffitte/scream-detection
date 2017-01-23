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
from theano.tensor.shared_randomstreams import RandomStreams
from io_func.model_io import _nnet2file, _cfg2file, _file2nnet, _cnn2file, _file2cnn
from utils.utils import parse_arguments
from utils.learn_rates import _lrate2file, _file2lrate
from utils.network_config import NetworkConfig
from learning.sgd import train_sgd, validate_by_minibatch
from models.cnn import CNN
from util_func import log

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['train_data', 'valid_data', 'nnet_spec', 'conv_nnet_spec', 'wdir']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']
    valid_data_spec = arguments['valid_data']
    conv_nnet_spec = arguments['conv_nnet_spec']
    nnet_spec = arguments['nnet_spec']
    wdir = arguments['wdir']
    multi_label = arguments['multi_label']
    if multi_label=="True": multi_label = True
    else:   multi_label = False

    # parse network configuration from arguments, and initialize data reading
    cfg = NetworkConfig(multi_label); cfg.model_type = 'CNN'
    cfg.parse_config_cnn(arguments, '10:' + nnet_spec, conv_nnet_spec)
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
    log('> ... initializing the model')
    # construct the cnn architecture
    cnn = CNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)
    # load the pre-training networks, if any, for parameter initialization
    if (ptr_layer_number > 0) and (resume_training is False):
        _file2nnet(cnn.layers, set_layer_num = ptr_layer_number, filename = ptr_file)
    if resume_training:
        _file2nnet(cnn.layers, filename = wdir + '/nnet.tmp')

    # get the training, validation and testing function for the model
    log('> ... getting the finetuning functions')
    train_fn, valid_fn = cnn.build_finetune_functions(
                (cfg.train_x, cfg.train_y), (cfg.valid_x, cfg.valid_y),
                batch_size=cfg.batch_size)

    log('> ... finetuning the model')
    while (cfg.lrate.get_rate() != 0):
        # one epoch of sgd training
        train_error, pred, _ = train_sgd(train_fn, cfg)
        labels = cfg.train_sets.label_vec
        correct_number = 0.0
        confusion_matrix = numpy.zeros((cfg.n_outs, cfg.n_outs))
        class_occurrence = numpy.zeros((1,cfg.n_outs))
        for i in range(len(pred)):
            p_sorted = pred[i]
            if p_sorted == labels[i]:
                correct_number += 1
                confusion_matrix[labels[i], labels[i]] += 1
            else:
                confusion_matrix[labels[i], p_sorted] += 1
            class_occurrence[0, labels[i]] += 1
        confusion_matrix = 100 * confusion_matrix / class_occurrence.T
        log('-->> Epoch %d, training error %f ' % (cfg.lrate.epoch, 100 * numpy.mean(train_error)) + '(%)')
        log('Confusion Matrix is \n\n ' + str(numpy.around(confusion_matrix, 2)) + ' (%)\n')
        # validation
        valid_error, pred2, _ = validate_by_minibatch(valid_fn, cfg)
        labels = cfg.valid_sets.label_vec
        correct_number = 0.0
        confusion_matrix = numpy.zeros((cfg.n_outs, cfg.n_outs))
        class_occurrence = numpy.zeros((1,cfg.n_outs))
        for i in range(len(pred2)):
            p_sorted = pred2[i]
            if p_sorted == labels[i]:
                correct_number += 1
                confusion_matrix[labels[i], labels[i]] += 1
            else:
                confusion_matrix[labels[i], p_sorted] += 1
            class_occurrence[0, labels[i]] += 1
        confusion_matrix = 100 * confusion_matrix / class_occurrence.T
        error_rate = 100 * (1.0 - correct_number / pred2.shape[0])
        log('-->> Epoch %d, lrate %f, validation error %f ' % (cfg.lrate.epoch, cfg.lrate.get_rate(),
                                                            100 * numpy.mean(valid_error)) + '(%)')
        log('Confusion Matrix: \n\n ' + str(numpy.around(confusion_matrix, 2)) + ' (%)\n')
        cfg.lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))
        # output nnet parameters and lrate, for training resume
        if cfg.lrate.epoch % cfg.model_save_step == 0:
            _nnet2file(cnn.layers, filename=wdir + '/nnet.tmp')
            _lrate2file(cfg.lrate, wdir + '/training_state.tmp')

    # save the model and network configuration
    print("param output file is:", cfg.param_output_file)
    if cfg.param_output_file != '':
        _nnet2file(cnn.layers, filename=cfg.param_output_file, input_factor = cfg.input_dropout_factor, factor = cfg.dropout_factor)
        log('> ... the final PDNN model parameter is ' + cfg.param_output_file)
    print("config output file is:", cfg.cfg_output_file)
    if cfg.cfg_output_file != '':
        _cfg2file(cnn.cfg, filename=cfg.cfg_output_file)
        log('> ... the final PDNN model config is ' + cfg.cfg_output_file)

    # output the fully-connected part into Kaldi-compatible format
    if cfg.kaldi_output_file != '':
        cnn.fc_dnn.write_model_to_kaldi(cfg.kaldi_output_file)
        log('> ... the final Kaldi model (only FC layers) is ' + cfg.kaldi_output_file)

    # remove the tmp files (which have been generated from resuming training)
    if os.path.isfile(wdir + '/nnet.tmp'):
        os.remove(wdir + '/nnet.tmp')
    if os.path.isfile(wdir + '/training_state.tmp'):
        os.remove(wdir + '/training_state.tmp')

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

# validation on the valid data; this involves a forward pass of all the valid data into the network,
# mini-batch by mini-batch
# valid_fn: the compiled valid function
# valid_sets: the dataset object for valid
# valid_xy: the tensor variables for valid dataset
# batch_size: the size of mini-batch
# return: a list containing the *error rates* on each mini-batch
import numpy as np

def validate_by_minibatch_verbose(valid_fn, valid_sets, valid_xy, batch_size):
    valid_error = []
    while (not valid_sets.is_finish()):
        valid_sets.load_next_partition(valid_xy)
        for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
            valid_error.append(valid_fn(index=batch_index))
    valid_sets.initialize_read()
    return valid_error


def validate_by_minibatch(valid_fn, cfg):
    valid_sets = cfg.valid_sets
    valid_xy = cfg.valid_xy
    batch_size = cfg.batch_size
    valid_error = []
    labels = []
    batch_size = cfg.batch_size
    index = 0
    prediction = []

    if cfg.model_type == 'RNN':
        while (not valid_sets.is_finish()):
            valid_sets.load_next_partition(valid_xy)
            seq_size = valid_sets.cur_frame_num
            error_value = valid_fn(index=index, size=seq_size)
            valid_error.append(error_value)

    else:
        while (not valid_sets.is_finish()):
            valid_sets.load_next_partition(valid_xy)
            for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
                error_value, pred, label = valid_fn(index=batch_index)
                valid_error.append(error_value)
                labels.append(label)
                if cfg.multi_label is True:
                    prediction.append(pred)
                else:
                    prediction = np.concatenate((prediction, pred))
    valid_sets.initialize_read()
    return valid_error, prediction, labels

# one epoch of mini-batch based SGD on the training data
# train_fn: the compiled training function
# train_sets: the dataset object for training
# train_xy: the tensor variables for training dataset
# batch_size: the size of mini-batch
# learning_rate: learning rate
# momentum: momentum
# return: a list containing the *error rates* on each mini-batch


def train_sgd_verbose(train_fn, train_sets, train_xy, batch_size, learning_rate, momentum):
    train_error = []
    while (not train_sets.is_finish()):
        train_sets.load_next_partition(train_xy)
        for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
            train_error.append(train_fn(index=batch_index, learning_rate=learning_rate, momentum=momentum))
    train_sets.initialize_read()
    return train_error


def train_sgd(train_fn, cfg):
    train_sets = cfg.train_sets
    train_xy = cfg.train_xy
    batch_size = cfg.batch_size
    index = 0
    learning_rate = cfg.lrate.get_rate()
    momentum = cfg.momentum
    train_error = []
    labels = []
    prediction = []

    if cfg.model_type == 'RNN':
        while (not train_sets.is_finish()):
            train_sets.load_next_partition(train_xy)
            seq_size = train_sets.cur_frame_num
            error_value, pred = train_fn(index=index, size=seq_size, learning_rate=learning_rate,
                                         momentum=momentum)
            train_error.append(error_value)
            prediction.append(pred)

    else:
        while (not train_sets.is_finish()):
            train_sets.load_next_partition(train_xy)
            for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
                error_value, pred, label = train_fn(index=batch_index, learning_rate=learning_rate,
                                             momentum=momentum)
                train_error.append(error_value)
                labels.append(label)
                if cfg.multi_label is True:
                    prediction.append(pred)
                else:
                    prediction = np.concatenate((prediction, pred))

    train_sets.initialize_read()
    return train_error, prediction, labels

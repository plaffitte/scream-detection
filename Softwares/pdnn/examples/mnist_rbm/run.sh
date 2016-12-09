#!/bin/bash

# two variables you need to set
pdnndir=/home/piero/Documents/Softwares/pdnn  # pointer to PDNN
theanodir=/home/piero/Theano
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export PYTHONPATH=$PYTHONPATH:$theanodir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

# download mnist dataset
#wget http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

# split the dataset to training, validation and testing sets
# you will see train.pickle.gz, valid.pickle.gz, test.pickle.gz
echo "Preparing datasets ..."
python data_prep.py

# train the RBM model
echo "Training the RBM model ..."
python $pdnndir/cmds/run_RBM.py --train-data "train.pickle.gz" \
                                --nnet-spec "784:128:128:10" --wdir ./ \
                                --epoch-number 10 --batch-size 128 --first_layer_type gb \
                                --ptr-layer-number 2 \
                                --param-output-file rbm.param --cfg-output-file rbm.cfg  >& rbm.training.log || exit 1;



#!/bin/bash

# two variables you need to set
pdnndir=/home/piero/Documents/Softwares/pdnn  # pointer to PDNN
theanodir=/home/piero/Theano
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export PYTHONPATH=$PYTHONPATH:$theanodir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

echo "Preparing datasets ..."
python prep_data.py

# train DNN model
echo "Training the DNN model ..."
python $pdnndir/cmds/run_DNN.py --train-data "train.pickle.gz" \
                                --valid-data "valid.pickle.gz" \
                                --nnet-spec "13:512:512:2" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:200" --model-save-step 20 \
                                --param-output-file dnn.param --cfg-output-file dnn.cfg  >& dnn.training.log

#python $pdnndir/cmds/run_DNN.py --train-data "/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/#wav/1421675810481/1421675810481.pickle" \
#                                --valid-data "/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan #wav/1421675810481/1421675810481.pickle" \
#                                --nnet-spec "13:1024:1024:1024:1024:5" --wdir ./ \
#                                --l2-reg 0.0001 --lrate "C:0.1:200" --model-save-step 20 \
#                                --param-output-file dnn.param --cfg-output-file dnn.cfg  >& dnn.training.log || exit 1;

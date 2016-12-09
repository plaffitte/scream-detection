#!/bin/bash

# two variables you need to set
pdnndir=/home/piero/Documents/Softwares/pdnn  # pointer to PDNN
theanodir=/home/piero/Theano
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export PYTHONPATH=$PYTHONPATH:$theanodir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

# The RBM model and the DNN model are composed in the way that the first 2 hidden layers are initialized with the RBM model. 
# These 2 layers remain unchanged during fine-tuning by specifying "--non-updated-layers"
echo "Training the DNN model ..."
python $pdnndir/cmds/run_DNN.py --train-data "train.pickle.gz" \
                                --valid-data "valid.pickle.gz" \
                                --nnet-spec "784:128:128:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:200" --model-save-step 20 \
                                --ptr-layer-number 2 --ptr-file rbm.param --non-updated-layers "0,1" \
                                --param-output-file dnn.param --cfg-output-file dnn.cfg  >& dnn.training.log || exit 1;

# classification on the test set
echo "Do classification on the test set ..."
python $pdnndir/cmds/run_Extract_Feats.py --data "test.pickle.gz" \
                                          --nnet-param dnn.param --nnet-cfg dnn.cfg \
                                          --output-file "dnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >& dnn.testing.log || exit 1;

python show_results.py dnn.classify.pickle.gz

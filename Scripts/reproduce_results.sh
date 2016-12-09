#!/bin/bash
# two variables you need to set
pdnndir=/home/piero/Documents/Softwares/pdnn  # pointer to PDNN
theanodir=/usr/local/lib/python2.7/dist-packages/theano
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs
classes="{Noise,Conversation,Shout}"
# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export PYTHONPATH=$PYTHONPATH:$theanodir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,OMP_NUM_THREADS=4,openmp='True'

path=$1
path=~/$path

# classification on the test set
echo "Do classification on the test set ..."
python $pdnndir/cmds/run_Extract_Feats.py --data "data/test.pickle.gz" \
                                          --nnet-param dnn.param --nnet-cfg dnn.cfg \
                                          --output-file "data/dnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 > dnn.testing.log;

python $path/show_results.py --pred_file "data/dnn.classify.pickle.gz" \
           --classes $classes \
           --filepath $path > control_test_results.log

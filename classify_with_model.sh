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
control_test_path=$2
control_test_path=/home/piero/Documents/test_folder/control_test_data/$control_test_path

python $control_test_path/control_test.py --data_type $2_test --target_path $path --initial_path $control_test_path

# classification on the test set
echo "Do classification on the test set ..."
python $pdnndir/cmds/run_Extract_Feats.py --data $path/$2_test.pickle.gz\
                                          --nnet-param $path/dnn.param --nnet-cfg $path/dnn.cfg \
                                          --output-file $path/$2_test_classify.pickle.gz --layer-index -1 \
                                          --batch-size 100 > $path/$2_test.dnn.testing.log;

python ~/Documents/Scripts/control_show_results.py --pred_file $path/$2_test_classify.pickle.gz \
           --classes $classes \
           --filepath $path > $path/$2_test_results.log


python /home/piero/Documents/Scripts/control_show_results_with_smoothing.py --pred_file $path/$2_test_classify.pickle.gz \
           --classes $classes \
           --filepath $path > $path/$2_test_results_test.log

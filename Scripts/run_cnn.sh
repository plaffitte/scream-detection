#!/bin/bash
pythoncmd="nice -n19  /usr/bin/python -u"
pdnndir=/home/piero/Documents/Softwares/pdnn  # pointer to PDNN
theanodir=/usr/local/lib/python2.7/dist-packages/theano
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs
export PYTHONPATH=$PYTHONPATH:$pdnndir
export PYTHONPATH=$PYTHONPATH:$theanodir
export PYTHONPATH=$PYTHONPATH:~/Documents/Scripts/
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,OMP_NUM_THREADS=2,openmp='True'

######################## FEATURES PARAMS ##############################
classes="{Noise,Conversation,Shouting}"
window_step=0.010 # in seconds, hop size between two successive mfcc windows
window_size=0.025 # in seconds, size of MFCC window
highfreq=20000 # maximal analysis frequency for mfcc
lowfreq=50 # minimal analysis frequency for mfcc
nfilt=40 # number of mfsc coef
N=50 # contextual window
slide=5
exp_path="/home/piero/Documents/Speech_databases/test/test_CNN" # path which experiment is run from
threshold=10000
compute_deltas="False"

######################## NETWORK PARAMS ##############################
cnn_architecture="1x50x40:150,8x10,p2x2:100,1x8,1x2,f"
dnn_architecture="512:3"
lambda="C:0.1:100"
l2=0.0
batch_size=100
rbm_epochs=100
rbm_layers=0

######################## CREATE DATASET ###############################
if [ "$(ls -A "$exp_path/data")" ] ; then
# if find /home/piero/Documents/Speech_databases/test/pickle_data -maxdepth 1 -mindepth 1 -name "*.log*" ; then
  echo 'data already prepared'
else
  echo "Preparing datasets ..."
  $pythoncmd /home/piero/Documents/Scripts/format_data_cnn_context.py --data_type "train" --classes $classes \
                --window_step $window_step --window_size $window_size --highfreq $highfreq --lowfreq $lowfreq \
                --nfilt $nfilt --N $N --slide $slide --exp_path $exp_path --threshold $threshold --deltas $compute_deltas
  $pythoncmd /home/piero/Documents/Scripts/format_data_cnn_context.py --data_type "test" --classes $classes \
                --window_step $window_step --window_size $window_size --highfreq $highfreq --lowfreq $lowfreq \
                --nfilt $nfilt --N $N --slide $slide --exp_path $exp_path --threshold $threshold --deltas $compute_deltas
  $pythoncmd /home/piero/Documents/Scripts/format_data_cnn_context.py --data_type "valid" --classes $classes \
                --window_step $window_step --window_size $window_size --highfreq $highfreq --lowfreq $lowfreq \
                --nfilt $nfilt --N $N --slide $slide --exp_path $exp_path --threshold $threshold --deltas $compute_deltas
fi

############################# TRAIN NETWORK ###########################
echo "Training the CNN model ..."
$pythoncmd $pdnndir/cmds/run_CNN.py --train-data "data/train.pickle.gz" --valid-data "data/valid.pickle.gz" \
                                --nnet-spec $dnn_architecture --wdir ./ --conv_nnet_spec $cnn_architecture --l2-reg $l2 --lrate $lambda \
                                 --model-save-step 10 --ptr-layer-number $rbm_layers --ptr_file "rbm.param" --batch-size $batch_size \
                                 --param-output-file cnn.param --cfg-output-file cnn.cfg > cnn.training.log;

echo "Do classification on the test set ..."
$pythoncmd $pdnndir/cmds/run_Extract_Feats.py --data "data/test.pickle.gz" \
                                          --nnet-param cnn.param --nnet-cfg cnn.cfg \
                                          --output-file "cnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 > cnn.testing.log;

$pythoncmd show_results.py --pred_file "cnn.classify.pickle.gz" \
           --classes $classes \
           --filepath "/home/piero/Documents/Speech_databases/test/test_CNN" > results.log

#!/bin/bash
## Display start time
#--------------------
echo
START=$(date +%s)
echo Simulation starts at: `date`
MODE_TEST_DEBUG=$1
namethisfic=`basename "$0"`
curdir="$( cd "$( dirname "${BASH_SOURCE[0] }" )" && pwd )"
# two variables you need to set
pdnndir=$curdir/Software/pdnn  # pointer to PDNN
device=cpu
pythoncmd="nice -n19  /usr/bin/python -u"
rnndir=$curdir/Software/RNN_LSTM
latexdir=$curdir/interface_latex
export PYTHONPATH=$PYTHONPATH:$pdnndir
export PYTHONPATH=$PYTHONPATH:$curdir/Scripts
export PYTHONPATH=$PYTHONPATH:$rnndir
export PYTHONPATH=$PYTHONPATH:$latexdir

######################## FEATURES PARAMS ##############################
classes="{Conversation,Shouting,Noise,BG_voice,Making_Of,Clavier,Compressor_Start,Laugh,Journal,Scream}"
rep_classes=`echo $classes | sed -e 's/,/_/g' -e 's/.//;s/.$//'`
typeparam="MFCC"
window_step=0.010 # in seconds, hop size between two successive mfcc windows
window_size=0.025 # in seconds, size of MFCC window
wst=$(echo "$window_step * 1000" |bc -l)
wsi=$(echo "$window_size * 1000" |bc -l)
window_steptxt=${wst/.000/m}
window_sizetxt=${wsi/.000/m}
highfreq=20000 # maximal analysis frequency for mfcc
lowfreq=50 # minimal analysis frequency for mfcc
size=40 # number of mfcc coef
#exp_path="$curdir/test_rnn" # path which experiment is run from
threshold=10000
max_seq_length=1000 #500
n_stream=5
max_batch_length=1000 #500
compute_deltas=False

# ################### NETWORK PARAMS ###########################
Nlayers=1                 # number of layers
Ndirs=1                     # unidirectional or bidirectional
Nh=56    4                # hidden layer size
Ah="lstm"                 # hidden unit activation (e.g. relu, tanh, lstm)
Ay="softmax"            # output unit activation (e.g. linear, sigmoid, softmax)
predictPer="frame"   # frame or sequence
loss="ce_group"         # loss function (e.g. mse, ce, ce_group, hinge, squared_hinge)
L1reg=0.001                # L1 regularization
L2reg=0.001                # L2 regularization
momentum=0.5         # SGD momentum
frontEnd="None"          # a lambda function for transforming the input
filename="None"          # initialize from file
initParams="None"       # initialize from given dic
n_epoch=1
lrate=0.05

couchetxt=$Nlayers"_layer_"$Ah"_"$Ay"_"$loss"_"$perdictPer"_"$Nh"_"$Nc
param_test="{$window_step,$window_size,$highfreq,$lowfreq,$size,$threshold,$compute_deltas}"
param_txt=$typeparam'_'$window_steptxt'_'$window_sizetxt'_'$highfreq'_'$lowfreq'_'$size'_'$max_seq_length'_sequence'
DATE_BEGIN=`date +[%Y-%m-%d\ %H:%M:%S.%N]`

if [ $MODE_TEST_DEBUG -eq 1 ] ; then
    echo " !!! MODE DEBUG - TOUT LE CONTNU DE $curdir/Features VA ETRE EFFACE !!! "
    echo " \\t CTRL + C pour arreter le processus et mettre MODE_TEST_DEBUG=0 dans run_rnn_yun.sh"
    echo " \\t N'importe que touche pour continuer"
##  read nimportequoi
fi

if [ $MODE_TEST_DEBUG -eq 1 ] ; then
    rm -rf $curdir/Features/*
fi

flag=0
if [ "$(ls $curdir/Features)" ] ; then
    if [ -d $curdir/Features/$rep_classes ]; then
        if [ -d $curdir/Features/$rep_classes/$param_txt ]; then
          echo "data already prepared..."
          dir_classes="$curdir/Features/$rep_classes"
          rep_classes="$dir_classes/$param_txt"
        else
          dir_classes="$curdir/Features/$rep_classes"
          rep_classes="$dir_classes/$param_txt"
          echo "creation  de " $rep_classes
          mkdir $rep_classes
          echo "Preparing datasets"
          cp $curdir/$namethisfic $rep_classes/$namethisfic
          flag=1
        fi
    else
      dir_classes="$curdir/Features/$rep_classes"
      echo "creation  de " $dir_classes
      mkdir $dir_classes
      rep_classes="$dir_classes/$param_txt"
      echo "creation  de " $rep_classes
      mkdir $rep_classes
      echo "Preparing datasets"
      cp $curdir/$namethisfic $rep_classes/$namethisfic
       flag=1
    fi
else
#  echo "Preparing datasets"
  dir_classes="$curdir/Features/$rep_classes"
  echo "creation  de " $dir_classes
  mkdir $dir_classes
  rep_classes="$dir_classes/$param_txt"
  echo "creation  de " $rep_classes
  mkdir $rep_classes
  cp $curdir/$namethisfic $rep_classes/$namethisfic
  echo "Preparing datasets"
  flag=1
fi

######################### CREATE DATASET ###############################
STEP1_START=$(date +%s)
if [ $flag -eq 1 ] ; then
  $pythoncmd $curdir/Scripts/format_data_rnn_yun_mix.py --data_type "train" --rep_test $rep_classes \
                        --param $param_test --classes $classes --max_seq_len $max_seq_length --n_stream $n_stream \
                        --max_batch_len $max_batch_length
  $pythoncmd $curdir/Scripts/format_data_rnn_yun_mix.py --data_type "valid" --rep_test $rep_classes \
                        --param $param_test --classes $classes --max_seq_len $max_seq_length --n_stream $n_stream \
                        --max_batch_len $max_batch_length
  $pythoncmd $curdir/Scripts/format_data_rnn_yun_mix.py --data_type "test" --rep_test $rep_classes \
                        --param $param_test --classes $classes --max_seq_len $max_seq_length --n_stream $n_stream \
                        --max_batch_len $max_batch_length
fi
STEP1_END=$(date +%s)

############################### TRAIN NETWORK ###########################
STEP2_START=$(date +%s)
if [ -d $rep_classes/$couchetxt ]; then
    echo "Model RNN/LSTM $couchetxt is detected with these data. Loading params"
    $pythoncmd $rnndir/run_rnn.py --train-data "$rep_classes/train.pickle.gz" --test-data "$rep_classes/test.pickle.gz" --valid-data "$rep_classes/valid.pickle.gz" \
                                    --nlayers $Nlayers --ndir $Ndirs --nx $size --nh $Nh --ah $Ah --ay $Ay --predict $predictPer --loss $loss \
                                    --l1 $L1reg --l2 $L2reg --momentum $momentum --frontEnd $frontEnd --filename $filename\
                                    --initparams $initParams --epoch $n_epoch --lambda $lrate > $rep_classes/$couchetxt/results.log \
                                    --filesave "$rep_classes/$couchetxt/rnn.params" --fileTex "$rep_classes/$couchetxt/tex" --classes $classes
else
    echo "Training the RNN model ..."
    mkdir "$rep_classes/$couchetxt"
    echo $param_test > $rep_classes/$couchetxt/log.txt
    mkdir "$rep_classes/$couchetxt/tex"
    $pythoncmd $rnndir/run_rnn.py --train-data "$rep_classes/train.pickle.gz" --test-data "$rep_classes/test.pickle.gz" --valid-data "$rep_classes/valid.pickle.gz" \
                                    --nlayers $Nlayers --ndir $Ndirs --nx $size --nh $Nh --ah $Ah --ay $Ay --predict $predictPer --loss $loss \
                                    --l1 $L1reg --l2 $L2reg --momentum $momentum --frontEnd $frontEnd --filename $filename\
                                    --initparams $initParams --epoch $n_epoch --lambda $lrate > $rep_classes/$couchetxt/results.log \
                                    --filesave "$rep_classes/$couchetxt/rnn.params" --fileTex "$rep_classes/$couchetxt/tex" --classes $classes
fi
STEP2_END=$(date +%s)

## Display end time
#------------------
echo
STOP=$(date +%s)
echo Simulation ends at `date `

## Compute and display elapsed time
#----------------------------------
echo
DIFF=$(( $STOP - $START ))
DAYS=$(( $DIFF / (60*60*24) ))
HR=$(( ($DIFF - $DAYS*60*60*24) / (60*60) ))
MIN=$(( ($DIFF - $DAYS*60*60*24 - $HR*60*60) / (60) ))
SEC=$(( $DIFF - $DAYS*60*60*24 - $HR*60*60 - $MIN*60 ))
echo "Elapsed time :: $DAYS jours $HR heures $MIN minutes et $SEC secondes"
echo

echo "$HOSTNAME $OMP_NUM_THREADS $(( $STEP1_END - $STEP1_START )) $(( $STEP2_END - $STEP2_START )) $(( $STOP - $START ))" >> results.log

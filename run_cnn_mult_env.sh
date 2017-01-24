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
export PYTHONPATH=$PYTHONPATH:$pdnndir
export PYTHONPATH=$PYTHONPATH:$curdir/Scripts
export PYTHONPATH=$PYTHONPATH:$curdir/interface_latex
device=cpu
pythoncmd="nice -n19  /usr/bin/python -u"

######################## FEATURES PARAMS ##############################
classes_1="{NO_SPEECH,SPEECH,SHOUT}"
classes_2="{START_UP_SPEED,STOP_SPEED,INCREASING_SPEED,FULL_SPEED,WORKSHOP_SPEED,SLOW_SPEED}"
typeparam="MFCC"
window_step=0.010 # in seconds, hop size between two successive mfcc windows
window_size=0.025 # in seconds, size of MFCC window
highfreq='20000' # maximal analysis frequency for mfcc
lowfreq='50' # minimal analysis frequency for mfcc
size='40' # number of mfcc or spectral coef
N='10' #contextual window
slide='5'
threshold=10000
compute_deltas="False"
cnn_arch="1x10x40:150,8x10,p2x2:100,1x8,1x2,f"
coucheDNN="512:18"
layerNumberDNN='1'
lambda="D:0.1:0.8:0.2,0.05:50" #"D:0.1:0.8:0.2,0.05" #
multi_label=false
multi_env=true

###################################################################
if $multi_label || $multi_env; then
  if $multi_env; then rep_classes="multi_env_"
  fi
  if $multi_label; then rep_classes="multi_lab_"
  fi
  rep_classes+=`echo $classes_1 | sed -e 's/,/_/g' -e 's/.//;s/.$//'`
  rep_classes+="x"
  rep_classes+=`echo $classes_2 | sed -e 's/,/_/g' -e 's/.//;s/.$//'`
else
  rep_classes=`echo $classes | sed -e 's/,/_/g' -e 's/.//;s/.$//'`
fi
wst=$(echo "$window_step * 1000" |bc -l)
wsi=$(echo "$window_size * 1000" |bc -l)
window_steptxt=${wst/.000/m}
window_sizetxt=${wsi/.000/m}
cnn_arch_txt=`echo $cnn_arch | sed -e 's/,/_/g'`
coucheCNNtxt=$cnn_arch_txt
coucheDNNtxt=$layerNumberDNN"x"$coucheDNN
param_test="{$window_step,$window_size,$highfreq,$lowfreq,$size,$N,$slide,$threshold,$compute_deltas,$multi_label}"
param_txt=$typeparam'_'$window_steptxt'_'$window_sizetxt'_'$highfreq'_'$lowfreq'_'$size'_'$N'_'$slide

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,openmp='True'

if [ $MODE_TEST_DEBUG -eq 1 ] ; then
    echo " !!! MODE DEBUG - TOUT LE CONTENU DE $curdir/Features VA ETRE EFFACE !!! "
    echo " \\t CTRL + C pour arreter le processus et mettre MODE_TEST_DEBUG=0 dans rbm_audio_classification.sh"
    echo " \\t N'importe quelle touche pour continuer"
##    read nimportequoi
fi
if [ $MODE_TEST_DEBUG -eq 1 ] ; then
    rm -rf $curdir/Features/*
fi
flag=0
dir_classes="$curdir/Features/$rep_classes"
int_dir="$dir_classes/$param_txt"
rep_classes="$int_dir/$coucheCNNtxt-$coucheDNNtxt"
if [ "$(ls $curdir/Features)" ] ; then
    if [ -d $dir_classes ]; then
        if [ -d $int_dir ]; then
          if [ -d $rep_classes ]; then
            echo "data already prepared..."
          else
            mkdir $rep_classes
            flag=1
          fi
        else
          echo "creation  de " $int_dir
          mkdir $int_dir
          mkdir $rep_classes
          echo "Preparing datasets"
          cp $curdir/$namethisfic $int_dir/$namethisfic
          flag=1
        fi
    else
      echo "creation  de " $dir_classes
      mkdir $dir_classes
      echo "creation  de " $rep_classes
      mkdir $int_dir
      echo "Preparing datasets"
      cp $curdir/$namethisfic $int_dir/$namethisfic
      mkdir $rep_classes
       flag=1
    fi
else
#  echo "Preparing datasets"
  echo "creation  de " $dir_classes
  mkdir $dir_classes
  echo "creation  de " $rep_classes
  mkdir $int_dir
  cp $curdir/$namethisfic $int_dir/$namethisfic
  mkdir $rep_classes
  echo "Preparing datasets"
  flag=1
fi

######################## CREATE DATASET ###############################
echo $param_test > $rep_classes/log.txt
echo $rep_classes
STEP1_START=$(date +%s)
if [ $flag -eq 1 ] ; then
    python $curdir/Scripts/format_data_mult_env_merge.py --data_type "train" --rep_test $rep_classes \
                        --param $param_test --classes_1 $classes_1 --classes_2 $classes_2 --net "CNN"
    python $curdir/Scripts/format_data_mult_env_merge.py --data_type "test" --rep_test $rep_classes \
                        --param $param_test --classes_1 $classes_1 --classes_2 $classes_2 --net "CNN"
    python $curdir/Scripts/format_data_mult_env_merge.py --data_type "valid" --rep_test $rep_classes \
                        --param $param_test --classes_1 $classes_1 --classes_2 $classes_2 --net "CNN"
fi
STEP1_END=$(date +%s)

######################## TRAIN CNN ###############################
if [ -d $rep_classes ]; then
    echo "Model DBN $coucheCNNtxt-$coucheDNNtxt exist."
    echo "The DBN model will be loaded"
else
    mkdir "$rep_classes"
    cp $curdir/$namethisfic $rep_classes/$namethisfic
fi
echo "Training the CNN model ..."
STEP3_START=$(date +%s)
python $pdnndir/cmds/run_CNN.py --train-data "$rep_classes/train.pickle.gz" --multi_label $multi_label \
                                --valid-data "$rep_classes/valid.pickle.gz" --conv_nnet_spec $cnn_arch\
                                --nnet-spec $coucheDNN --wdir $rep_classes \
                                --l2-reg 0.001 --lrate $lambda --model-save-step 1 --ptr-layer-number 0 \
                                --param-output-file  $rep_classes/cnn.param \
                                --cfg-output-file $rep_classes/cnn.cfg \
                                > $rep_classes/cnn.training.log
STEP3_END=$(date +%s)

######################## CLASSIFICATION ON TEST DATASET ###############################
echo "Do classification on the test set ..."
STEP4_START=$(date +%s)
python $pdnndir/cmds/run_Extract_Feats.py --data "$rep_classes/test.pickle.gz" \
                                          --nnet-param  $rep_classes/cnn.param \
                                          --nnet-cfg  $rep_classes/cnn.cfg --layer-index -1 \
                                          --output-file "$rep_classes/cnn.classify.pickle.gz"  \
                                          --batch-size 100 >  $rep_classes/cnn.testing.log;
STEP4_END=$(date +%s)

mkdir $rep_classes/tex
STEP5_START=$(date +%s)
echo $rep_classes
python $curdir/Scripts/show_results.py --pred_file "$rep_classes/cnn.classify.pickle.gz" \
           --datapath $rep_classes --classes $classes --filepath $rep_classes/tex --nametex "Confusion_matrix"
STEP5_END=$(date +%s)

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
echo "$HOSTNAME $OMP_NUM_THREADS $(( $STEP1_END - $STEP1_START )) $(( $STEP2_END - $STEP2_START )) $(( $STEP3_END - $STEP3_START )) $(( $STEP4_END - $STEP4_START )) $(( $STEP5_END - $STEP5_START )) $(( $STOP - $START ))" >> results.log

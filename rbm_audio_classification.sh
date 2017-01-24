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
export PYTHONPATH=$PYTHONPATH:$pdnndir
export PYTHONPATH=$PYTHONPATH:$curdir/Scripts
export PYTHONPATH=$PYTHONPATH:$curdir/interface_latex
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,openmp='True'

############################### SET PARAMS #############################
# For regular single environment classification
classes="{Noise,Conversation,Shouting}"
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
ptr=0 # set to 1 if you want pre-training
coucheRBM=""
coucheDNN="400:128:3"
layerNumberRBM='0'
layerNumberDNN='1'
epoch_numberRBM=10
lambda="C:0.1:1" # "D:0.1:0.8:0.2,0.05:10"
multi_label=false
## source $1

###################################################################
if $multi_label; then
  rep_classes=`echo $classes_1 | sed -e 's/,/_/g' -e 's/.//;s/.$//'`
  rep_classes+="x"
  rep_classes+=`echo $classes_2 | sed -e 's/,/_/g' -e 's/.//;s/.$//'`
else
  rep_classes=`echo $classes | sed -e 's/,/_/g' -e 's/.//;s/.$//'`
fi
wst=$(echo "$window_step * 1000" |bc -l)
wsi=$(echo "$window_size * 1000" |bc -l)
window_steptxt=${wst/.000/m}
window_sizetxt=${wsi/.000/m}
coucheRBMtxt=$layerNumberRBM"x"$coucheRBM
coucheDNNtxt=$layerNumberDNN"x"$coucheDNN
param_test="{$window_step,$window_size,$highfreq,$lowfreq,$size,$N,$slide,$threshold,$compute_deltas}"
param_txt=$typeparam'_'$window_steptxt'_'$window_sizetxt'_'$highfreq'_'$lowfreq'_'$size'_'$N'_'$slide

if [ $MODE_TEST_DEBUG -eq 1 ] ; then
    echo " !!! MODE DEBUG - TOUT LE CONTENU DE $curdir/Features VA ETRE EFFACE !!! "
    echo " \\t CTRL + C pour arreter le processus et mettre MODE_TEST_DEBUG=0 dans rbm_audio_classification.sh"
    echo " \\t N'importe quelle touche pour continuer"
    rm -rf $curdir/Features/*
##    read nimportequoi
fi
flag=0
dir_classes="$curdir/Features/$rep_classes"
int_dir="$dir_classes/$param_txt"
dnn_rep="$int_dir/$coucheRBMtxt"
rep_classes="$dnn_rep/$coucheDNNtxt"
if [ "$(ls $curdir/Features)" ] ; then
    if [ -d $dir_classes ]; then
        if [ -d $int_dir ]; then
          if [ -d $dnn_rep ]; then
            if [ -d $rep_classes ]; then
              echo "data already prepared..."
            else
              mkdir $rep_classes
              flag=1
            fi
          else
            mkdir $dnn_rep
            mkdir $rep_classes
            flag=1
          fi
        else
          echo "creation  de " $int_dir
          mkdir $int_dir
          mkdir $dnn_rep
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
      mkdir $dnn_rep
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
  mkdir $dnn_rep
  mkdir $rep_classes
  echo "Preparing datasets"
  flag=1
fi
######################## CREATE DATASET ###############################
echo $param_test > $rep_classes/log.txt
if [ -d $dnn_rep ]; then
    echo "Model RBM $coucheRBMtxt exists."
    echo "The RBM model will be loaded"
else
    mkdir $dnn_rep
    cp $curdir/$namethisfic $dnn_rep/$namethisfic
fi
if [ -d $rep_classes ]; then
    echo "Model DBN $coucheRBMtxt/$coucheDNNtxt exist."
    echo "The DBN model will be loaded"
else
    mkdir $rep_classes
    cp $curdir/$namethisfic $rep_classes/$namethisfic
fi

STEP1_START=$(date +%s)
if [ $flag -eq 1 ] ; then
    python $curdir/Scripts/format_pickle_data.py --data_type "train" --rep_test $rep_classes\
                                                                  --param $param_test --classes $classes
    python $curdir/Scripts/format_pickle_data.py --data_type "valid" --rep_test $rep_classes\
                                                                   --param $param_test --classes $classes
    python $curdir/Scripts/format_pickle_data.py --data_type "test" --rep_test $rep_classes\
                                                                   --param $param_test --classes $classes
fi
STEP1_END=$(date +%s)
STEP2_START=$(date +%s)
######################## PRE TRAINING ###############################
if [ "$ptr" -ne 0 ]; then
  echo "Pre-training the model ..."
  python $pdnndir/cmds/run_RBM.py --train_data "$rep_classes/train.pickle.gz" --nnet_spec $coucheRBM \
                                  --wdir $dnn_rep --epoch_number $epoch_numberRBM --batch_size 128 \
                                  --first_layer_type gb --ptr_layer_number $layerNumberRBM \
                                  --param_output_file $dnn_rep/rbm.param --multi_label $multi_label\
                                  --cfg_output_file $dnn_rep/rbm.cfg  > $dnn_rep/rbm.training.log
fi
STEP2_END=$(date +%s)
######################## TRAIN DNN ###############################
echo "Discriminatively fine-tuning the DNN model ..."
STEP3_START=$(date +%s)
python $pdnndir/cmds/run_DNN.py --train-data "$rep_classes/train.pickle.gz" \
                                --valid-data "$rep_classes/valid.pickle.gz" --multi_label $multi_label\
                                --nnet-spec $coucheDNN --wdir $rep_classes --l2-reg 0.0 --lrate $lambda --model-save-step 10 \
                                --batch_size 128 --ptr-layer-number $layerNumberRBM --ptr-file $dnn_rep/rbm.param \
                                --param-output-file  $rep_classes/dnn.param --cfg-output-file $rep_classes/dnn.cfg > $rep_classes/dnn.training.log
STEP3_END=$(date +%s)
######################## CLASSIFICATION ON TEST DATASET ###############################
echo "Do classification on the test set ..."
STEP4_START=$(date +%s)
python $pdnndir/cmds/run_Extract_Feats.py --data "$rep_classes/test.pickle.gz" \
                                          --nnet-param  $rep_classes/dnn.param --nnet-cfg  $rep_classes/dnn.cfg \
                                          --output-file "$rep_classes/dnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >  $rep_classes/dnn.testing.log;
STEP4_END=$(date +%s)
mkdir $rep_classes/tex
STEP5_START=$(date +%s)
echo $rep_classes
python $curdir/Scripts/show_results.py --pred_file "$rep_classes/dnn.classify.pickle.gz" \
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

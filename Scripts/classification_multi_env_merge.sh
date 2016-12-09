#!/bin/bash

## Display start time
#--------------------
echo
START=$(date +%s)
echo Simulation starts at: `date`

MODE_TEST_DEBUG=1

##namethisfic="rbm_audio_classification.sh"
namethisfic=`basename "$0"`
curdir="$( cd "$( dirname "${BASH_SOURCE[0] }" )" && pwd )"
echo $curdir
# two variables you need to set
pdnndir=$curdir/Software/pdnn  # pointer to PDNN
device=cpu

#Paramètre du test
classes="{Conversation,Shouting,Noise}"
classes_1="{NO_SPEECH,SPEECH,SHOUT}"
classes_2="{PARKED,START_UP_SPEED,REDUCTION_SPEED,FULL_SPEED,SLOW_SPEED,STOP_SPEED}"
rep_classes=`echo $classes_1 | sed -e 's/,/_/g' -e 's/.//;s/.$//'``echo $classes_2 | sed -e 's/,/_/g' -e 's/.//;s/.$//'`
typeparam="MFCC"
window_step=0.010 # in seconds, hop size between two successive mfcc windows
window_size=0.025 # in seconds, size of MFCC window
wst=$(echo "$window_step * 1000" |bc -l)
wsi=$(echo "$window_size * 1000" |bc -l)
window_steptxt=${wst/.000/m}
window_sizetxt=${wsi/.000/m}
highfreq='20000' # maximal analysis frequency for mfcc
lowfreq='50' # minimal analysis frequency for mfcc
size='40' # number of mfcc coef
N='10' #contextual window
slide='5'

coucheRBM=""
coucheDNN="400:256:3"
layerNumberRBM='0'
layerNumberDNN='1'
coucheRBMtxt=$layerNumberRBM"_layer_RBM_0"
coucheDNNtxt=$layerNumberDNN"_layer_DNN_256_3"

epoch_numberRBM=0
epoch_numberDNN=5

param_test="{$window_step,$window_size,$highfreq,$lowfreq,$size,$N,$slide}"
param_txt=$typeparam'_'$window_steptxt'_'$window_sizetxt'_'$highfreq'_'$lowfreq'_'$size'_'$N'_'$slide

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,openmp='True'
# export OMP_NUM_THREADS=2
# export PATH=/home/piero/etc/miniconda2/bin:$PATH
# source activate test_pierre

if [ $MODE_TEST_DEBUG -eq 1 ] ; then
    echo " !!! MODE DEBUG - TOUT LE CONTENU DE $curdir/Features VA ETRE EFFACE !!! "
    echo " \\t CTRL + C pour arreter le processus et mettre MODE_TEST_DEBUG=0 dans rbm_audio_classification.sh"
    echo " \\t N'importe que touche pour continuer"
##    read nimportequoi
fi
if [ $MODE_TEST_DEBUG -eq 1 ] ; then
    rm -rf $curdir/Features/*
fi
flag=0
if [ "$(ls $curdir/Features)" ] ; then
    if [ -d $curdir/Features/$rep_classes ]; then
        if [ -d $curdir/Features/$rep_classes/$param_txt ]; then
          echo "echo 'data already prepared...'"
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

echo $param_test > $rep_classes/log.txt
echo $rep_classes
STEP1_START=$(date +%s)
if [ $flag -eq 1 ] ; then
python $curdir/Scripts/format_pickle_data.py --data_type "train" --path $curdir --rep_test $rep_classes --param $param_test --classes $classes --N $Nc
    python $curdir/Scripts/format_pickle_data.py --data_type "test" --path $curdir --rep_test $rep_classes --param $param_test --classes $classes --N $Nc
    python $curdir/Scripts/format_pickle_data.py --data_type "valid" --path $curdir --rep_test $rep_classes --param $param_test --classes $classes --N $Nc
fi
STEP1_END=$(date +%s)

if [ -d $rep_classes/$coucheRBMtxt ]; then
    echo "Model RBM $coucheRBMtxt exists."
    echo "The RBM model will be loaded"
else
    mkdir "$rep_classes/$coucheRBMtxt"
    cp $curdir/$namethisfic $rep_classes/$coucheRBMtxt/$namethisfic
fi
STEP2_START=$(date +%s)
python $pdnndir/cmds/run_RBM.py --train_data "$rep_classes/train.pickle.gz" --nnet_spec $coucheRBM \
                                --wdir $rep_classes/$coucheRBMtxt --epoch_number $epoch_numberRBM --batch_size 128 \
                                --first_layer_type gb --ptr_layer_number $layerNumberRBM \
                                --param_output_file $rep_classes/$coucheRBMtxt/rbm.param \
                                --cfg_output_file $rep_classes/$coucheRBMtxt/rbm.cfg  > $rep_classes/$coucheRBMtxt/rbm.training.log
STEP2_END=$(date +%s)

# train DNN model
if [ -d $rep_classes/$coucheRBMtxt/$coucheDNNtxt ]; then
    echo "Model DBN $coucheRBMtxt/$coucheDNNtxt exist."
    echo "The DBN model will be loaded"
else
    mkdir "$rep_classes/$coucheRBMtxt/$coucheDNNtxt"
    cp $curdir/$namethisfic $rep_classes/$coucheRBMtxt/$coucheDNNtxt/$namethisfic
fi
echo "Training the DNN model ..."
STEP3_START=$(date +%s)
python $pdnndir/cmds/run_DNN.py --train-data "$rep_classes/train.pickle.gz" \
                                --valid-data "$rep_classes/valid.pickle.gz" \
                                --nnet-spec $coucheDNN --wdir $rep_classes/$coucheRBMtxt/$coucheDNNtxt \
                                --l2-reg 0.001 --lrate "C:0.1:$epoch_numberDNN" --model-save-step 10 \
                                --ptr-layer-number $layerNumberRBM --ptr-file $rep_classes/$coucheRBMtxt/rbm.param \
                                --param-output-file  $rep_classes/$coucheRBMtxt/$coucheDNNtxt/dnn.param --cfg-output-file $rep_classes/$coucheRBMtxt/$coucheDNNtxt/dnn.cfg  > $rep_classes/$coucheRBMtxt/$coucheDNNtxt/dnn.training.log
STEP3_END=$(date +%s)

# classification on the test set
echo "Do classification on the test set ..."
STEP4_START=$(date +%s)
python $pdnndir/cmds/run_Extract_Feats.py --data "$rep_classes/test.pickle.gz" \
                                          --nnet-param  $rep_classes/$coucheRBMtxt/$coucheDNNtxt/dnn.param --nnet-cfg  $rep_classes/$coucheRBMtxt/$coucheDNNtxt/dnn.cfg \
                                          --output-file "$rep_classes/$coucheRBMtxt/$coucheDNNtxt/dnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >  $rep_classes/$coucheRBMtxt/$coucheDNNtxt/dnn.testing.log;
STEP4_END=$(date +%s)


mkdir $rep_classes/$coucheRBMtxt/$coucheDNNtxt/tex
STEP5_START=$(date +%s)
python show_results.py --pred_file "$rep_classes/$coucheRBMtxt/$coucheDNNtxt/dnn.classify.pickle.gz" \
           --datapath $rep_classes \
           --classes $classes \
           --filepath $rep_classes/$coucheRBMtxt/$coucheDNNtxt/tex\
           --nametex "Confusion_matrix"
STEP5_END=$(date +%s)

#cd $rep_classes/$coucheRBMtxt/$coucheDNNtxt/tex
#pdflatex Confusion_matrix.tex
#cp Confusion_matrix.pdf $rep_classes/$coucheRBMtxt/$coucheDNNtxt/Confusion_matrix.pdf
#cp Confusion_matrix.tex $rep_classes/$coucheRBMtxt/$coucheDNNtxt/tex/Confusion_matrix.tex
#cp table.tex $rep_classes/$coucheRBMtxt/$coucheDNNtxt/tex/table.tex
#rm Confusion_matrix.pdf
#rm Confusion_matrix.tex
#rm Confusion_matrix.aux
#rm Confusion_matrix.log
#rm table.tex
#rm table.aux

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
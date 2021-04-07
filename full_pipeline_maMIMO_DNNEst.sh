#!/bin/bash

#Arguments:
#
#

MATLAB=matlab20
MAT_CODEDIR=packet_generation/phased_arr
MODEL_DIR=magazine_review/BS32_denoise_3k_SNR120
PYDATASET_DIR=datasets_maMIMO/SNRanalysis
Nt=32
Nr=4
TRAIN_Npkt=3000	# tot. num of transmissions in training set
TEST_Npkt=500	# for each SNR level

mkdir -p $MAT_CODEDIR/packets/SNRanalysis

# 1) generate train/test matlab files
#test
echo "Starting test set generation in bg.."
cd $MAT_CODEDIR
for s in -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -5 0 5 10
do
  echo screen -dmS "matGen_SNR$s" $MATLAB -nodesktop -nosplash -r "addpath('/home/mauro/Documents/MATLAB/Examples/R2020a/phased_comm/MassiveMIMOHybridBeamformingExample'); generate_maMIMO_LTF(strcat('1UsrTest_BS32_SNR',num2str($s)),$TEST_Npkt,$Nt,$Nr,$s,false,true,false,[],true); exit;"
done
# while we generate test data in background (NOTE, they should have less Npkt than training set), generate training dataset 
#train
echo "Generating training set.."
echo $MATLAB -nodesktop -nosplash -r "addpath('/home/mauro/Documents/MATLAB/Examples/R2020a/phased_comm/MassiveMIMOHybridBeamformingExample'); generate_maMIMO_LTF(strcat('1UsrTrain_BS32_SNR',num2str(120)),$TRAIN_Npkt,$Nt,$Nr,$s,false,true,false,[],true); exit;"
cd ../../

# 2) generate train and test datasets for python scripts
echo "Generating Python ready dataset files form matlab format.."
# test data
for s in -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -5 0 5 10
do
  echo screen -dmS "pyDataConvert_SNR$s" python create_massiveMIMO_CSIest_dnn_dataset.py -x $MAT_CODEDIR/packets/SNRanalysis/maMIMO_$TEST_Npkt___1UsrTest_BS32_SNR$s.mat -o $PYDATASET_DIR/SNRanalysis/testDataset$TEST_Npkt_1Usr_BS32_SNR$s.b
done
# train data
echo python create_massiveMIMO_CSIest_dnn_dataset.py -x $MAT_CODEDIR/packets/SNRanalysis/maMIMO_$TRAIN_Npkt___1Usr_BS32_SNR120.mat -o $PYDATASET_DIR/dataset$TRAIN_Npkt_1Usr_BS32_SNR120_NOISELESS.b

# 3) train
echo "Start training the model.."
echo python massiveMIMO_CSI_prediction_DNN.py --train -x $PYDATASET_DIR/dataset$TRAIN_Npkt_1Usr_BS32_SNR120_NOISELESS.b --nn 1024 1024 -d $MODEL_DIR --bs 256 --epochs 1000 --method default_SNR --useGPU 0 --useBN --datasource matlab_maMimo

# 4) generate test output
echo "Generate test output."
for s in -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -5 0 5 10
do
  mkdir -p $MODEL_DIR/test_results/BS32_SNR$s  # test script doesn't create folders automatically
  echo python massiveMIMO_CSI_prediction_DNN.py --test -x $PYDATASET_DIR/testDataset$TEST_Npkt_1Usr_BS32_SNR$s.b --nn 1024 1024 -d $MODEL_DIR/test_results/BS32_SNR$s --modeldir $MODEL_DIR --useGPU 0 --useBN --datasource matlab_maMimo --valSameTrain
done

#5) produce output plots TODO: double check this part
cd $MAT_CODEDIR
echo $MATLAB -nodesktop -nosplash -r "addpath('/home/mauro/Documents/MATLAB/Examples/R2020a/phased_comm/MassiveMIMOHybridBeamformingExample'); snr_loop_testing; exit;"
cd ../../

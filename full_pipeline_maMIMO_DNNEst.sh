#!/bin/bash

# load environment variables
source setenv.sh

mkdir -p $MAT_CODEDIR/packets/SNRanalysis

if [ $IS_GEN_DATA -eq 1 ]
then
  # 1) generate train/test matlab files
  #test
  echo "Starting test set generation in bg.."
  cd $MAT_CODEDIR
  for s in $SNRLev
  do
      screen -dmS "matGen_SNR$s" $MATLAB -nodesktop -nosplash -r "addpath('$MMIMO_BF_EX_DIR'); generate_maMIMO_LTF(strcat('1UsrTest_BS${Nt}_SNR',num2str($s)),$TEST_Npkt,$Nt,$Nr,$s,true,true,false,[],false); exit;"
  done
  # while we generate test data in background (NOTE, they should have less Npkt than training set), generate training dataset
  #train
  echo "Generating training set.."
  $MATLAB -nodesktop -nosplash -r "addpath('$MMIMO_BF_EX_DIR'); generate_maMIMO_LTF(strcat('1UsrTrain_BS${Nt}_SNR',num2str(120)),$TRAIN_Npkt,$Nt,$Nr,120,true,true,false,[],false); exit;"
  cd ../../

  mkdir -p $PYDATASET_DIR/SNRanalysis
  # 2) generate train and test datasets for python scripts
  echo "Generating Python ready dataset files form matlab format.."
  # test data
  for s in $SNRLev
  do
      screen -dmS "pyDataConvert_SNR$s" $PY create_massiveMIMO_CSIest_dnn_dataset.py -x $MAT_CODEDIR/packets/SNRanalysis/maMIMO_${TEST_Npkt}___1UsrTest_BS${Nt}_SNR${s}.mat -o $PYDATASET_DIR/SNRanalysis/testDataset${TEST_Npkt}_1Usr_BS${Nt}_SNR${s}.b
  done
  # train data
  $PY create_massiveMIMO_CSIest_dnn_dataset.py -x $MAT_CODEDIR/packets/SNRanalysis/maMIMO_${TRAIN_Npkt}___1UsrTrain_BS${Nt}_SNR120.mat -o $PYDATASET_DIR/SNRanalysis/dataset${TRAIN_Npkt}_1Usr_BS${Nt}_SNR120_NOISELESS.b
fi

mkdir -p $MODEL_DIR/BS${Nt}_denoise_${TRAIN_Npkt}_SNR120
# 3) train
echo "Start training the model.."
$PY massiveMIMO_CSI_prediction_DNN.py --train -x $PYDATASET_DIR/SNRanalysis/dataset${TRAIN_Npkt}_1Usr_BS${Nt}_SNR120_NOISELESS.b --nn 1024 1024 -d $MODEL_DIR/BS${Nt}_denoise_${TRAIN_Npkt}_SNR120 --bs 256 --epochs 1000 --method default_SNR --useGPU 0 --useBN --datasource matlab_maMimo

# 4) generate test output
echo "Generate test output."
for s in $SNRLev
do
  mkdir -p $MODEL_DIR/test_results/BS${Nt}_SNR$s  # test script doesn't create folders automatically
  $PY massiveMIMO_CSI_prediction_DNN.py --test -x $PYDATASET_DIR/SNRanalysis/testDataset${TEST_Npkt}_1Usr_BS${Nt}_SNR$s.b --nn 1024 1024 -d $MODEL_DIR/test_results/BS${Nt}_SNR$s --modeldir $MODEL_DIR --useGPU 0 --useBN --datasource matlab_maMimo --valSameTrain
done

cd $MAT_CODEDIR
# 5) generate the output metrics for testing the model performance
for s in $SNRLev
do
    screen -dmS "mtestMetricsGen_SNR$s" $MATLAB -nodesktop -nosplash -r "addpath('$MMIMO_BF_EX_DIR'); BER_test_maMIMO_LTF('packets/SNRanalysis/maMIMO_${TEST_Npkt}___1UsrTest_BS${Nt}_SNR${s}.mat', ${TEST_Npkt}, 1.0, '../../$MODEL_DIR/test_results/BS${Nt}_SNR${s}/test_csi_predictions_real.mat', '../../$MODEL_DIR/test_results/BS${Nt}_SNR${s}/test_csi_predictions_imag.mat',false,true,$s); exit;"
done

# 6) produce output plots
$MATLAB -nodesktop -nosplash -r "addpath('$MMIMO_BF_EX_DIR'); snr_loop_testing('packets/SNRanalysis','../../$MODEL_DIR/test_results', $TEST_Npkt, [$SNRLev], $Nt); exit;"

cd ../../


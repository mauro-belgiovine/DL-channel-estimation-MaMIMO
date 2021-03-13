#!/bin/bash
#Arguments:
#	$1 - working directory, used to save models
#

MATLAB=matlab20
MAT_CODEDIR=packet_generation/phased_arr
MODEL_DIR=magazine_review/BS32_denoise_3k_SNR120
PYDATASET_DIR=datasets_maMIMO/SNRanalysis

# 1) generate train/test matlab files
#train
cd $MAT_CODEDIR
#TODO: add script to generate training data
cd ../../
#test
cd $MAT_CODEDIR
# TODO: THIS VERSION WHERE WE LOAD THE PARAMETERS OBJECT (prm) IN MATLAB FOR SOME REASON IS NOT ABLE TO DECODE THE PACKETS
# ALSO, WHEN THE PARFOR IS USED, THE INITIAL SEED IS DIFFERENT. THE PARAMETER GENERATION PART MUST BE REWRITTEN, 
# HENCE DO NOT USE PARFOR AND GENERATE EACH RUN INDIVIDUALLY
#matlab20 -nodesktop -nosplash -r "addpath('/home/mauro/Documents/MATLAB/Examples/R2020a/phased_comm/MassiveMIMOHybridBeamformingExample'); snr_loop; exit;"
for s in -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -5 0 5 10
do
  screen -dmS "matGen_SNR$s" $MATLAB -nodesktop -nosplash -r "addpath('/home/mauro/Documents/MATLAB/Examples/R2020a/phased_comm/MassiveMIMOHybridBeamformingExample'); generate_maMIMO_LTF(strcat('1UsrTest_BS32_SNR',num2str($s)),500,32,4,$s,false,true,false,[],true); exit;"
done
cd ../../
# 2) generate train and test datasets for python
# train data
python create_massiveMIMO_CSIest_dnn_dataset.py -x $MAT_CODEDIR/packets/SNRanalysis/maMIMO_3000___1Usr_BS32_SNR120.mat -o $PYDATASET_DIR/dataset3K_1Usr_BS32_SNR120_NOISELESS.b
# test data
for s in -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -5 0 5 10
do
  screen -dmS "pyDataConvert_SNR$s" python create_massiveMIMO_CSIest_dnn_dataset.py -x $MAT_CODEDIR/packets/SNRanalysis/maMIMO_500___1UsrTest_BS32_SNR$s.mat -o $PYDATASET_DIR/SNRanalysis/testDataset500_1Usr_BS32_SNR$s.b
done

# 3) train 
python massiveMIMO_CSI_prediction_DNN.py --train -x $PYDATASET_DIR/dataset3K_1Usr_BS32_SNR120_NOISELESS.b --nn 1024 1024 -d $MODEL_DIR --bs 256 --epochs 1000 --method default_SNR --useGPU 1 --useBN --datasource matlab_maMimo

# 4) generate test output
for s in -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -5 0 5 10
do
  mkdir -p $MODEL_DIR/test_results/BS32_SNR$s  # test script doesn't create folders automatically
  python massiveMIMO_CSI_prediction_DNN.py --test -x $PYDATASET_DIR/testDataset500_1Usr_BS32_SNR$s.b --nn 1024 1024 -d $MODEL_DIR/test_results/BS32_SNR$s --modeldir $MODEL_DIR --useGPU 0 --useBN --datasource matlab_maMimo --valSameTrain
done

#5) produce output plots TODO: double check this part
cd packet_generation/phased_arr/
#matlab20 -nodesktop -nosplash -r "addpath('/home/mauro/Documents/MATLAB/Examples/R2020a/phased_comm/MassiveMIMOHybridBeamformingExample'); snr_loop_testing; exit;"
#for s in -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -5 0 5 10
#do
#  sbatch ~/run_cpu_bash.sh matlab -nodesktop -nosplash -r "nPkts=500;testData_path='/scratch/belgiovine.m/CSI_estimation/$1/test_results/';BER_test_maMIMO_LTF(strcat('packets/SNRanalysis/maMIMO_',num2str(nPkts),'___1Usr_BS32_SNR',num2str($s),'.mat'),nPkts,1.0,strcat(testData_path,'BS32_SNR',num2str($s),'/test_csi_predictions_real.mat'),strcat(testData_path,'BS32_SNR',num2str($s),'/test_csi_predictions_imag.mat'),false,true,$s);exit;" 
#done
cd ../../

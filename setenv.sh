#!/bin/bash
# main paths
PY=~/anaconda3/envs/py3_tf2.3/bin/python
MATLAB=matlab20
MAT_CODEDIR=packet_generation/phased_arr
MODEL_DIR=magazine_review/BS32_denoise_3k_SNR120
PYDATASET_DIR=datasets_maMIMO
MMIMO_BF_EX_DIR=~/Documents/MATLAB/Examples/R2020a/phased_comm/MassiveMIMOHybridBeamformingExample

# simulation parameters
Nt=32
Nr=4

# this is the original configuration of the paper. NOTE: this will run for a long time and require a lot of resources.
#TRAIN_Npkt=9000	# tot. num of transmissions in training set
#TEST_Npkt=500	# for each SNR level
#SNRLev="-22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -5 0 5 10"  # SNR levels considered

# reduced number of transmissions (for testing purpose)
TRAIN_Npkt=1000	# tot. num of transmissions in training set
TEST_Npkt=250	# for each SNR level
SNRLev="-22 -20 -15 -10 -5 0 5 10"  # SNR levels considered
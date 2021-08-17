# Deep Learning at the Edge for Channel Estimation in Beyond-5G Massive MIMO
This repository contains the code needed to reproduce results in the paper  by M. Belgiovine, et al. *“Deep Learning at the Edge for Channel Estimation in Beyond-5G Massive MIMO”*, accepted at IEEE Wireless Communications Magazine (WCM),  April 2021. The [manuscript](https://ieeexplore.ieee.org/abstract/document/9430899) is available on IEEE Xplore.

Please cite our work if using this code for your experiments:
```
@article{Belgiovine2021DeepLA,
  title={Deep Learning at the Edge for Channel Estimation in Beyond-5G Massive MIMO},
  author={M. Belgiovine and Kunal Sankhe and Carlos Bocanegra and D. Roy and K. Chowdhury},
  journal={IEEE Wireless Communications},
  year={2021},
  volume={28},
  pages={19-25}
}
```


## Dataset Download
The Training and Testing dataset to reproduce the paper results can be downloaded on [Genesys Website](https://genesys-lab.org/CS-5g-beyond). The web page also offer additional insights on the dataset created and the obtained performance.

## How to run (Ubuntu 18.04)
### Dependencies
Data generation requires a lot of resources, a machine with multicore and lots of RAM is recommended. Also, for TensorFlow, a GPU is recommended.
Software requirements:
- Python 3.7
- TensorFlow (2.3)
- Matlab (2020b)
- Matlab Phased Antenna Array Toolbox and [Hybrid Beamforming Example](https://www.mathworks.com/help/phased/ug/massive-mimo-hybrid-beamforming.html) installed on your machine
### Configure environment
First configure simulation parameters in [setenv.sh](setenv.sh). Here is an example of a simulation configuration:
```
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
TRAIN_Npkt=3000	# tot. num of transmissions in training set
TEST_Npkt=500	# for each SNR level
SNRLev="-25 -20 -15 -10 -5 0 5 10"  # SNR levels considered
```
Is it possible to define different SNR levels to be evaluated, number of transmitter and receiver antennas, number of training and test samples. Note that increasing the values of simulation parameters will requires more resources.

### Run the whole simulation
Run the script [full_pipeline_maMIMO_DNNEst.sh](full_pipeline_maMIMO_DNNEst.sh) to:
- generate the Training and Testing datasets through Matlab
- train the DNN models for channel estimation
- test the trained models on test data
- generate output performance

To run the script, type the following in your terminal from the project root directory:
```
bash full_pipeline_maMIMO_DNNEst.sh
```
If you have already downloaded the [dataset](https://genesys-lab.org/CS-5g-beyond), you can skip the data generation portion and directly train the DNN models.

## Contacts
Please reach out to belgiovine.m@northeastern.edu for any question. 

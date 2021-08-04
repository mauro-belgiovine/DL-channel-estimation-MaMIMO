import argparse
parser = argparse.ArgumentParser(description='Generate dataset from .mat files.')
parser.add_argument('-x', help='Input datafile used for train/test. Use wildcards to select multiple files.')
parser.add_argument('-o', '--output', default='dataset.b', help='Dataset output pickle filename.')
parser.add_argument('--user', type=int, default=0, help="Indicate user # to extract data from (for multiuser scenario)")

args = parser.parse_args()

import numpy as np
import h5py
import glob
import pickle
import random

f_list = glob.glob(args.x)
trainLTF = {}
trainX = None # Initialized later
trainy_real = None
trainy_imag = None

# these indexes are consistent across old dataset loading function and new one
ix_ltf = 0
ix_CSI = 1

# file list is sorted to facilitate test phase in Matlab and recover data from single .mat files
for i,x in enumerate(sorted(f_list)):

    print('Processing file '+str(i+1)+'/'+str(len(f_list)))
    with h5py.File(x) as f:

        data_X = f[(f['usr_data'][ix_ltf, args.user])].value  # we read the complete matrix once, because it takes a lot of time
        data_y = f[(f['usr_data'][ix_CSI, args.user])].value

        nUsers = f['usr_data'].shape[1]
        nRxAts, inPreambLen, nPackets = data_X.shape
        nRxAts, nTxAts, nSubCarr, nPackets = data_y.shape
        P = f['P'].value # retrieve pilot sequences matrix

        trainX_temp = np.zeros((nPackets*nTxAts*nRxAts, 2), dtype=int) # for every datapoint, this will contain every unique LTF relative [hash, iTx]
        trainy_real_temp = np.zeros((nPackets*nRxAts*nTxAts,nSubCarr))
        trainy_imag_temp = np.zeros((nPackets*nRxAts*nTxAts,nSubCarr))



        for p in range(nPackets):
            print('Transmission '+str(p))

            for iRx in range(nRxAts):
                print('Rx atx '+str(iRx)+'...')
                # instead of saving the same sequence for all 64 tx antennas,
                # we just save its unique hash and save the LTF in a dictionary
                while True:
                    hash = random.getrandbits(32)
                    if not(hash in trainLTF):
                        obj_ltf = data_X[iRx, :, p]
                        unzip_ltf = list(zip(*obj_ltf))
                        trainLTF[hash] = {  'real': np.asarray(unzip_ltf[0]),
                                            'imag': np.asarray(unzip_ltf[1])}
                        break

                for iTx in range(nTxAts):
                    samp_ix = p*(nRxAts*nTxAts)+iRx*(nTxAts)+iTx
                    trainX_temp[samp_ix] = [hash, iTx]

                    unzip_csi = list(zip(*data_y[iRx, iTx, :, p]))
                    trainy_real_temp[samp_ix] = unzip_csi[0]
                    trainy_imag_temp[samp_ix] = unzip_csi[1]

                    """
                    trainX.append(
                        (hash,
                         iTx)
                    )
                    trainy.append(
                        f[(f['usr_data'][ix_CSI, 0])].value[iRx, iTx, :, p]
                    )
                    """
        # retrieve additional simulation parameters
        # TODO: consider how to collect this with multiple file
        fftLen = f['prm']['FFTLength'].value[0,0]
        cpLen = f['prm']['CyclicPrefixLength'].value[0,0]
        simParams = {
            'FFTLength': fftLen,
            'CPLen': cpLen,
            'numSym': inPreambLen/(fftLen+cpLen),
            'symOffset': cpLen,
            'nTX': nTxAts,
            'nRX': nRxAts
        }

    # handle concatenation of data from different files
    if trainX is None:
        trainX = trainX_temp
    else:
        trainX = np.concatenate((trainX, trainX_temp))

    if (trainy_real is None) and (trainy_imag is None):
        trainy_real = trainy_real_temp
        trainy_imag = trainy_imag_temp
    else:
        trainy_real = np.concatenate((trainy_real,trainy_real_temp))
        trainy_imag = np.concatenate((trainy_imag,trainy_imag_temp))

"""
# convert trainX and trainy in numpy arrays
trainX = np.asarray(trainX)
trainy = np.asarray(trainy)
print(len(trainX), len(trainy))

trainLTF = {} # output dictionary
trainy_real = np.zeros((nPackets, nSubCarr,))
trainy_imag = np.zeros((nPackets, nSubCarr,))

for k in trainLTF_temp.keys():
    unzip_LTF = list(zip(*trainLTF_temp[k]))
    trainLTF[k] = {'real': unzip_LTF[0],
                   'imag': unzip_LTF[1]}

for p in range(nPackets):
    unzip_trainy = list(zip(*trainy[0]))
    trainy_real = unzip_trainy[0]
    trainy_imag = unzip_trainy[1]
"""

dataset = {'X': trainX, 'y': {'real': trainy_real, 'imag': trainy_imag}, 'LTF': trainLTF, 'P': P, 'simParams': simParams }
pickle.dump(dataset, open(args.output, 'wb'))



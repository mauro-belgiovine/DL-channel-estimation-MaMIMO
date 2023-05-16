import numpy as np
import random
from tensorflow.keras.utils import Sequence
import pickle
#import multiprocessing

def rms(X):
    return (np.sqrt(np.mean(X * X))) # root mean square

def rms_complex(X):
    return np.sqrt(np.mean(X.real * X.real + X.imag * X.imag))



def loadDataset(args):

    dataset_val = None

    # retrieve informations from dataset
    if args.datasource == 'matlab_maMimo':

        with open(args.x, 'rb') as f:
            # load dataset from pickle file
            dataset = pickle.load(f)

        n_samples = dataset['X'].shape[0]
        ltf_len = dataset['LTF'][dataset['X'][0, 0]]['real'].shape[0]
        n_out = dataset['y']['real'].shape[1]
        args.nTX = dataset['simParams']['nTX']
        args.nRX = dataset['simParams']['nRX']

        dataset_norm = dataset
        #    dataset_norm['X']['real'] = preprocessing.normalize(dataset['X']['real'], norm='l2')
        #    dataset_norm['X']['real'] = preprocessing.normalize(dataset['X']['real'], norm='l2')

        simParams = dataset['simParams']
        simParams['lenLTF'] = ltf_len
        simParams['nSubCarr'] = n_out

        # import scipy.io as sio
        # ltf_freq = sio.loadmat(os.path.join(os.path.dirname(args.x),'ltf_freq.mat'))
        # simParams['ltf_freqdom'] = np.reshape(ltf_freq['ltf_f'],(ltf_freq['ltf_f'].shape[0],))

        for k in ['FFTLength', 'CPLen', 'numSym', 'symOffset']:
            simParams[k] = int(simParams[k])  # sanitize type for array indexing in data generator

        n_packets = n_samples / (args.nTX * args.nRX)
        if not n_packets.is_integer():
            print(
                'Num. of packets is not an integer. Please double check --nTX and --nRX arguments to match the provided dataset. Aborting...')
            exit(-1)
        n_test_packets = int(np.floor(n_packets * args.valTrainRatio))

        n_val_samples = n_test_packets * (args.nTX * args.nRX)  # last transmissions are left as test
        n_train_samples = n_samples - n_val_samples

    elif args.datasource == 'matlab_WiFi_SISO':

        with open(args.x, 'rb') as f:
            # load dataset from pickle file
            dataset = pickle.load(f)

        n_samples = dataset['X']['real'].shape[0]
        n_out = dataset['y']['real'].shape[1]
        # todo insert these value during dataset creation time
        simParams = {
            'FFTLength': 512,
            'CPLen': 128,
            'numSym': 1280 / (512 + 128),
            'symOffset': 128,
            'nTX': 1,
            'nRX': 1,
            'lenLTF': 1280,
            'nSubCarr': 416
        }

        # for SISO case, force not using Dropout or Batch Normalization.
        # maybe this will be useful if we increase the network size and training samples?
        # args.dropout = 0.0
        args.useBN = False

        n_val_samples = int(np.floor(n_samples * args.valTrainRatio))
        n_train_samples = n_samples - n_val_samples

    elif args.datasource == 'matlab_SpiNN_WiFi':

        with open(args.x, 'rb') as f:
            # load dataset from pickle file
            dataset = pickle.load(f)

        # in this case we use preamble in the frequency domain (after OFDM demodulation)
        n_samples = dataset['X']['real'].shape[0]
        n_in = dataset['X']['real'].shape[1]
        n_out = dataset['y']['real'].shape[1]
        simParams = {
            'FFTLength': 64,
            'numSym': 2,  # num. of OFDM symbos in L-LTF
            'nTX': 1,
            'nRX': 1,
            'nSubCarr': 52,
            'lenLTF': n_in
        }

        if args.y != '':
            # load validation input file
            with open(args.y, 'rb') as f:
                # load dataset from pickle file
                dataset_val = pickle.load(f)
            n_val_samples = dataset_val['X']['real'].shape[0]
            n_train_samples = n_samples
        else:
            n_val_samples = int(np.floor(n_samples * args.valTrainRatio))
            n_train_samples = n_samples - n_val_samples

    elif args.datasource == 'powder':

        with open(args.x, 'rb') as f:
            # load dataset from pickle file
            dataset = pickle.load(f)

        n_samples = dataset['X'].shape[0]
        n_out = dataset['y'].shape[1]
        simParams = dataset['simParams']

        # we don't have channels for all packets, so we treat each channel independently
        n_val_samples = int(np.floor(n_samples * args.valTrainRatio))
        n_train_samples = n_samples - n_val_samples

    elif args.datasource == 'RICE_RENEW':
        data_out = pickle.load(open(args.x, 'rb'))

        BSatx = data_out['chan_est'].shape[2]
        nFrames = data_out['chan_est'].shape[3]
        nCli = data_out['chan_est'].shape[1]
        nSampRX = data_out['lts_RX'].shape[4]
        FFTlen = data_out['chan_est'].shape[4]

        if FFTlen == 64:
            nonZero_subcarr_ix = list(range(6, 32)) + list(range(33, 59))
        # TODO handle other FFT size for non-null subcarriers
        n_out = FFTlen - (FFTlen - len(nonZero_subcarr_ix)) # this will be set based on the num of non-null subcarriers found in first usable packet

        dataset = {'X': {'real': None, 'imag': None},
                   'y': {'real': None, 'imag': None}}

        frame_map = np.zeros((nCli, nFrames), dtype=np.bool)

        n_samples = 0
        for cli in range(nCli):
            for f in range(nFrames):

                nonzeros = [np.count_nonzero(data_out['chan_est'][0, cli, a, f, :]) for a in range(BSatx)]
                for i, z in enumerate(nonzeros):
                    if z > 0:
                        frame_map[cli, f] = True

            # extract indeces of non-zero samples
            ixs = np.nonzero(frame_map[cli])[0]
            n_samples += len(ixs)*BSatx
            # retrieve input data
            Xtmp = [np.squeeze(data_out['lts_RX'][0, cli, a, ixs, :]) for a in range(BSatx)]
            Xtmp_concat = np.concatenate(Xtmp)
            ytmp = [np.squeeze(data_out['chan_est'][0, cli, a, ixs, :]) for a in range(BSatx)]
            ytmp_concat = np.concatenate(ytmp)

            if (dataset['X']['real'] is None) and (dataset['X']['imag'] is None) and (
                    dataset['y']['real'] is None) and (dataset['y']['imag'] is None):
                dataset['X']['real'] = Xtmp_concat.real
                dataset['X']['imag'] = Xtmp_concat.imag
                # retireve num. of usable subcarriers from first usable sample (it's assumed to be consistent across the dataset)
                # use fft shift for collected data, as this is not performed by the RICE script when saving data
                # also, remove zeros from arrays (i.e. null subcarriers) by only taking non-zero subcarrier indexes (after shift)
                dataset['y']['real'] = np.fft.fftshift(ytmp_concat.real, axes=1)[:,nonZero_subcarr_ix]
                dataset['y']['imag'] = np.fft.fftshift(ytmp_concat.imag, axes=1)[:,nonZero_subcarr_ix]
            else:
                dataset['X']['real'] = np.concatenate((dataset['X']['real'], Xtmp_concat.real))
                dataset['X']['imag'] = np.concatenate((dataset['X']['imag'], Xtmp_concat.imag))
                dataset['y']['real'] = np.concatenate((dataset['y']['real'], np.fft.fftshift(ytmp_concat.real, axes=1)[:,nonZero_subcarr_ix]))
                dataset['y']['imag'] = np.concatenate((dataset['y']['imag'], np.fft.fftshift(ytmp_concat.imag, axes=1)[:,nonZero_subcarr_ix]))

        simParams = {
            'FFTLength': 64,
            'numSym': 1,  # num. of OFDM symbos in L-LTF
            'nTX': 1,   # we treat each client independently in this case, as in the RICE-RENEW provided code (MMIMO_RECEIVER.py)
            'nRX': 64,
            'nSubCarr': n_out,
            'lenLTF': nSampRX
        }

        n_val_samples = int(np.floor(n_samples * args.valTrainRatio))
        n_train_samples = n_samples - n_val_samples

    elif args.datasource == 'Carlos-gnuradio':

        dataset = pickle.load(open(args.x, 'rb'))
        n_samples = dataset['X']['real'].shape[0]
        n_in = dataset['X']['real'].shape[1]
        n_out = dataset['y']['real'].shape[1]

        simParams = {
            'FFTLength': 64,
            'numSym': 1,  # num. of OFDM symbos in L-LTF
            'nTX': 1,
            'nRX': 1,
            'nSubCarr': n_out,
            'lenLTF': n_in
        }

        n_val_samples = int(np.floor(n_samples * args.valTrainRatio))
        n_train_samples = n_samples - n_val_samples

    if dataset_val is None:
        dataset_val = dataset   # if validation dataset is not set explicitly, is the same as training and will be splitted

    return dataset, n_samples, n_train_samples, n_val_samples, dataset_val, simParams

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dataset, d, simParams, datasource='matlab_maMimo', method='default', fraction=1, model='FC', batch_size=128, n_channels=1, shuffle=True):
        'Initialization'
        self.prm = simParams
        self.batch_size = batch_size
        self.dataset = dataset
        self.d = d # 'real' or 'imag'
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.method = method    # 'default' -> whole LTF + pilot sequence
                                # 'reshape' -> reshape as if in OFDM demodulation
        self.datasource = datasource    # 'matlab_maMimo' -> Data from Matlab simulation (default)
                                        # 'powder' -> Real transmission from powder
        self.model = model
        self.fraction = fraction
        #self.pool = multiprocessing.Pool(processes=32)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, rms_fact  = self.__data_generation(list_IDs_temp)

        return X, y, rms_fact

    def reorder_indexes(self):
        'Restore original sample order'
        self.indexes = np.arange(len(self.list_IDs))

    def set_batchsize(self, bs):
        self.batch_size = bs

    def get_batchsize(self):
        return self.batch_size

    def get_method(self):
        return self.method

    def get_datasource(self):
        return self.datasource

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    """
    def addNoise(self, sampleIx):
        rxSig = self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d][0:int(self.prm['lenLTF']/self.fraction)]
        # apply noise to desired SNR level.
        SNRs = [40, 20, 10, 0]
        SNR = random.choice(SNRs)  # desired SNR
        if SNR < 40:
            std = np.sqrt(  # standard deviation of noise
                (np.sqrt(np.mean(rxSig * rxSig))) / pow(10, SNR / 10)
            )

            # draw noise samples from normal dist. (scaled of sqrt(2) factor)
            noise_norm = np.random.normal(0, 1, rxSig.shape) / np.sqrt(2)
            output = rxSig + std * noise_norm  # apply noise to the signal
            # Store sample
        else:
            output = rxSig
        return output
    """
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        cur_batch_size = len(list_IDs_temp) # account for right batch size, when num. of samples is not divisible for self.batch_size
        rms_fact = None # only filled by mehtod that uses rms normalization (probably all, eventually)

        if self.datasource == 'matlab_maMimo':

            if (self.method == 'default') or (self.method == 'default_SNR'):

                Xsig = np.empty((cur_batch_size, int(self.prm['lenLTF'] / self.fraction),1))  # retrieve LTF HALF! int(self.prm['lenLTF']/2)
                Xp = np.empty((cur_batch_size, self.prm['nTX'],))
                y = np.empty((cur_batch_size, self.prm['nSubCarr']))
                # Generate data
                for i, sampleIx in enumerate(list_IDs_temp):
                    # Store sample
                    Xsig[i] = self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d][
                              0:int(self.prm['lenLTF'] / self.fraction)][:, np.newaxis]
                    Xp[i] = self.dataset['P'][:, self.dataset['X'][sampleIx, 1]]

                    # Store class
                    y[i] = self.dataset['y'][self.d][sampleIx, :]

                X = [Xsig, Xp]

                """
                if self.method == 'default':
                    # Initialization
                    if self.model == 'FC':
                        X = np.empty((cur_batch_size, int(self.prm['lenLTF']/self.fraction)+self.prm['nTX'],))                                     # retrieve LTF HALF! int(self.prm['lenLTF']/2)
                        #X = np.empty((cur_batch_size, self.prm['lenLTF'],))                            # without append P
                    elif self.model =='CONV1D':
                        X = np.empty((cur_batch_size, self.prm['lenLTF'] + self.prm['nTX'],1))     # CNN input
    
                    y = np.empty((cur_batch_size, self.prm['nSubCarr']))
    
                    # Generate data
                    for i,sampleIx in enumerate(list_IDs_temp):
                        # Store sample
                        if self.model == 'FC':
                            X[i] = np.append(self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d][0:int(self.prm['lenLTF']/self.fraction)],   # retrieve LTF HALF! int(self.prm['lenLTF']/2)
                                               self.dataset['P'][:, self.dataset['X'][sampleIx, 1]])  # append the preamble sequence
                            #X[i] = self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d] # without append P ---> doesn't work, as expected :)
    
                        elif self.model =='CONV1D':
    
                            #input_nn = np.append(self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d],
                            #          self.dataset['P'][:, self.dataset['X'][sampleIx, 1]])
    
                            input_nn = self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d]
    
                            X[i] = np.reshape(input_nn,                                 # reshape for CNN input
                                             (self.prm['lenLTF'] + self.prm['nTX'],1))
    
                        # Store class
                        y[i] = self.dataset['y'][self.d][sampleIx,:]
                elif self.method == 'default_SNR':
    
                    # Initialization
                    if self.model == 'FC':
                        Xsig = np.empty((cur_batch_size, int(self.prm['lenLTF'] / self.fraction),1))  # retrieve LTF HALF! int(self.prm['lenLTF']/2)
                        Xp = np.empty((cur_batch_size, self.prm['nTX'],))
                        # X = np.empty((cur_batch_size, self.prm['lenLTF'],))                            # without append P
                    elif self.model == 'CONV1D':
                        X = np.empty((cur_batch_size, self.prm['lenLTF'] + self.prm['nTX'], 1))  # CNN input
    
                    y = np.empty((cur_batch_size, self.prm['nSubCarr']))
    
                    # Generate data
                    for i, sampleIx in enumerate(list_IDs_temp):
                        # Store sample
                        if self.model == 'FC':
                            Xsig[i] = self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d][
                                             0:int(self.prm['lenLTF'] / self.fraction)][:, np.newaxis]
                            Xp[i] = self.dataset['P'][:,self.dataset['X'][sampleIx, 1]]
                                             # retrieve LTF HALF! int(self.prm['lenLTF']/2)
                                               # append the preamble sequence
                            # X[i] = self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d] # without append P ---> doesn't work, as expected :)
    
                        elif self.model == 'CONV1D':
    
                            input_nn = np.append(self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d],
                                      self.dataset['P'][:, self.dataset['X'][sampleIx, 1]])
    
                            #input_nn = self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d]
    
                            X[i] = np.reshape(input_nn,  # reshape for CNN input
                                              (self.prm['lenLTF'] + self.prm['nTX'], 1))
    
                        # Store class
                        y[i] = self.dataset['y'][self.d][sampleIx, :]
    
                    X =  [Xsig, Xp ]
                """

                """
                # Next code is part of self.method == 'default_SNR'
                # THIS CAN BE USEFUL TO PRODUCE MORE FINE GRAINED AWGN NOISE APPLICATION WHILE GENERATING DATA
                # ALTHOUGH, AS IS, IT IS VERY SLOW
                
                X = np.empty((cur_batch_size, int(self.prm['lenLTF'] / self.fraction) + self.prm[
                    'nTX'],))  # retrieve LTF HALF! int(self.prm['lenLTF']/2)
                y = np.empty((cur_batch_size, self.prm['nSubCarr']))
                # Generate data
                for i, sampleIx in enumerate(list_IDs_temp):
                    # Store sample
                    X[i] = np.append(self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d][
                                     0:int(self.prm['lenLTF'] / self.fraction)],
                                     # retrieve LTF HALF! int(self.prm['lenLTF']/2)
                                     self.dataset['P'][:,
                                     self.dataset['X'][sampleIx, 1]])  # append the preamble sequence
                    # Store class
                    y[i] = self.dataset['y'][self.d][sampleIx, :]
    
                # retrieve the incoming signal
    
                # apply noise to desired SNR level.
                SNRs = [40, 20, 10, 0]
                SNR = np.random.choice(SNRs, (cur_batch_size,))  # desired SNR
    
                noise_pow = (np.apply_along_axis(rms, 1, X[:, 0:self.prm['lenLTF']])) ** 2 / pow(10,
                                                                                                 SNR / 10)  # noise pow for all signals
                std = np.sqrt(noise_pow)  # standard deviation of noise (noise power is the variance)
    
                # draw noise samples from normal dist. (scaled of sqrt(2) factor)
                noise_norm = np.random.normal(0, 1, (X.shape[0], self.prm['lenLTF'])) / np.sqrt(2)
                X[:, 0:self.prm['lenLTF']] = X[:, 0:self.prm['lenLTF']] + np.multiply(noise_norm, std[:,
                                                                                                  np.newaxis])  # apply noise to the signal
    
                # NOTE: np.multiply is used to multiply each element of std (contains the std of noise elementwise) for every row of noise_norm matrix.
                """

            elif self.method == 'reshape':
                # THIS METHOD PERFORMS COMPLETE OFDM DEMODULATION

                # X = np.empty((cur_batch_size, int(self.prm['FFTLength']+self.prm['CPLen'])+self.prm['nTX'], ))    # without CP removal
                X = np.empty((cur_batch_size, int(self.prm['FFTLength']) + self.prm['nTX'] + self.prm['nSubCarr'],))                      # with CP removal
                y = np.empty((cur_batch_size, self.prm['nSubCarr']))

                # Generate data
                for i, sampleIx in enumerate(list_IDs_temp):
                    # Store sample
                    ltfSig = self.dataset['LTF'][self.dataset['X'][sampleIx, 0]][self.d]
                    # reshape the LTF signal to obtain each OFDM/Pilot symbol
                    input2D = np.reshape(ltfSig,
                                         (self.prm['FFTLength']+self.prm['CPLen'], self.prm['numSym']),
                                         order='F').copy() # NOTE matlab reshape function use Fortran ordering. I have checked and output matches Matlab reshape function!

                    # Cyclic prefix (CP) removal
                    noCP_ix = list(range(self.prm['CPLen'], self.prm['FFTLength'] + self.prm['symOffset'])) + list(range(self.prm['symOffset'], self.prm['CPLen']))
                    afterCPRemoval = input2D[noCP_ix, :]

                    pilot_seq = np.append(self.dataset['P'][:, self.dataset['X'][sampleIx, 1]], # NOTE: I checked if we retrieve the right pilot sequence as in Matlab
                                          self.prm['ltf_freqdom'])

                    #X[i] = np.append(afterCPRemoval[:, self.dataset['X'][sampleIx, 1]],
                    #                  pilot_seq)


                    afterFFT = np.fft.fft(afterCPRemoval, n=self.prm['FFTLength'], axis=0)  # FFT
                    afterFFT = np.fft.fftshift(afterFFT)    # shift output to have center frequency in the middle
                    X[i] = np.append(afterFFT[:,self.dataset['X'][sampleIx, 1]],
                                     pilot_seq)

                    # Store class
                    y[i] = self.dataset['y'][self.d][sampleIx,:]

        elif self.datasource == 'matlab_WiFi_SISO':

            if self.method == 'default' or self.method == 'default_SNR':
                # Initialization
                #X = np.empty((cur_batch_size, self.prm['lenLTF']+self.prm['nSubCarr'],))
                X = np.empty((cur_batch_size, self.prm['lenLTF'],))                            # without append P
                # X = np.empty((cur_batch_size, self.prm['lenLTF'] + self.prm['nTX'],1))     # CNN input
                y = np.empty((cur_batch_size, self.prm['nSubCarr']))

                # Generate data
                for i,sampleIx in enumerate(list_IDs_temp):
                    # Store sample
                    X[i] = self.dataset['X'][self.d][sampleIx, :]   # retrieve LTF
                    # Store class
                    y[i] =  self.dataset['y'][self.d][sampleIx,:]
            """
            elif self.method == 'default_SNR':

                # Initialization
                # X = np.empty((cur_batch_size, self.prm['lenLTF']+self.prm['nSubCarr'],))
                X = np.empty((cur_batch_size, self.prm['lenLTF'],))  # without append P
                # X = np.empty((cur_batch_size, self.prm['lenLTF'] + self.prm['nTX'],1))     # CNN input
                y = np.empty((cur_batch_size, self.prm['nSubCarr']))

                # Generate data
                for i, sampleIx in enumerate(list_IDs_temp):
                    # Store sample
                    X[i] = self.dataset['X'][self.d][sampleIx, :]  # retrieve LTF
                    # apply noise to desired SNR level
                    SNR = 20  # desired SNR
                    std = np.sqrt(  # standard deviation of noise
                        self.dataset['Xpow'][sampleIx] / pow(10, SNR / 10)
                    )
                    # draw noise samples from normal dist. (scaled of sqrt(2) factor)
                    noise_norm = np.random.normal(0, 1, X[i].shape) / np.sqrt(2)
                    X[i] = X[i] + std * noise_norm  # apply noise to the signal

                    # Store class
                    y[i] = self.dataset['y'][self.d][sampleIx, :]
            """

        elif self.datasource == 'matlab_SpiNN_WiFi' or self.datasource == 'Carlos-gnuradio':
            if self.method == 'default' or self.method == 'default_SNR':
                # Initialization
                X = np.empty((cur_batch_size, self.prm['lenLTF'],))
                y = np.empty((cur_batch_size, self.prm['nSubCarr']))
                # create an additional output for RMS normalization factor
                rms_fact = np.empty((cur_batch_size,), dtype=np.complex128)

                # Generate data
                for i,sampleIx in enumerate(list_IDs_temp):
                    # normalize signal power
                    # generate complex LTF signal
                    ltfsig = self.dataset['X']['real'][sampleIx, :] + 1j * self.dataset['X']['imag'][sampleIx, :]
                    # compute RMS
                    rms_fact[i] = rms_complex(ltfsig)
                    a = rms_complex(ltfsig)
                    ltfsig_norm = ltfsig / a
                    chanest = self.dataset['y']['real'][sampleIx, :] + 1j * self.dataset['y']['imag'][sampleIx, :]
                    chanest_norm = chanest / a

                    # Store sample
                    if self.d == 'real':
                        in_X = ltfsig_norm.real
                        in_y = chanest_norm.real
                    elif self.d == 'imag':
                        in_X =  ltfsig_norm.imag
                        in_y = chanest_norm.imag

                    X[i] =  in_X  # retrieve LTF in Freq. domain
                    # Store class
                    y[i] = in_y

        elif self.datasource == 'powder':
            if self.method == 'default':

                if self.d == 'real':
                    totX = self.dataset['X'].real
                    toty = self.dataset['y'].real
                else:
                    totX = self.dataset['X'].imag
                    toty = self.dataset['y'].imag

                # Initialization
                X = np.empty((cur_batch_size, self.prm['lenLTF']+self.prm['nSubCarr'],))
                #X = np.empty((cur_batch_size, self.prm['lenLTF'],))                            # without append P
                # X = np.empty((cur_batch_size, self.prm['lenLTF'] + self.prm['nTX'],1))     # CNN input
                y = np.empty((cur_batch_size, self.prm['nSubCarr']))

                # Generate data
                for i,sampleIx in enumerate(list_IDs_temp):
                    # Store sample
                    X[i] = np.append(totX[sampleIx, :],   # retrieve LTF
                                       self.dataset['P'])  # append the preamble sequence

                    # Store class
                    y[i] = toty[sampleIx,:]

        elif self.datasource == 'RICE_RENEW':
            if self.method == 'default':
                if self.d == 'real':
                    totX = self.dataset['X']['real']
                    toty = self.dataset['y']['real']
                else:
                    totX = self.dataset['X']['imag']
                    toty = self.dataset['y']['imag']

                # Initialization
                X = np.empty((cur_batch_size, self.prm['lenLTF'],))
                # X = np.empty((cur_batch_size, self.prm['lenLTF'],))                            # without append P
                # X = np.empty((cur_batch_size, self.prm['lenLTF'] + self.prm['nTX'],1))     # CNN input
                y = np.empty((cur_batch_size, self.prm['nSubCarr']))

                # Generate data
                for i, sampleIx in enumerate(list_IDs_temp):
                    # Store sample
                    X[i] = totX[sampleIx, :]  # append the preamble sequence

                    # Store class
                    y[i] = toty[sampleIx, :]


        """elif self.datasource == 'Carlos-gnuradio':
            if self.method == 'default' or self.method == 'default_SNR':
                # Initialization
                X = np.empty((cur_batch_size, self.prm['lenLTF'],))
                y = np.empty((cur_batch_size, self.prm['nSubCarr']))

                # Generate data
                for i, sampleIx in enumerate(list_IDs_temp):
                    # Store sample
                    X[i] = self.dataset['X'][self.d][sampleIx, :]  # retrieve LTF
                    # Store class
                    y[i] = self.dataset['y'][self.d][sampleIx, :]"""


        return X, y, rms_fact
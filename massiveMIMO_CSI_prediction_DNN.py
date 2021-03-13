
import argparse

parser = argparse.ArgumentParser(description='Train-test CSI prediction network')
me_group = parser.add_mutually_exclusive_group()
me_group.add_argument('--train', help='Trigger training of the network on train data', action='store_true')
me_group.add_argument('--test', help='Trigger testing of the network on test data', action='store_true')
me_group.add_argument('--input_opt', help='It loads a model and performs optimization finding the input that maximizes one neuron activation.')
parser.add_argument('--model', default='FC', help='Specify DNN model type. Possible values are: FC, CONV1D.')
parser.add_argument('-x', required=True, help='Input datafile used for train/test')
parser.add_argument('-y', default='', help='Input datafile used for test. If not specified, test is created from "-x" argument using the last samples.')
parser.add_argument('--datasource', required=True, default='matlab_maMimo', help='Specify the data source in order to setup Data Generator accordingly.')
parser.add_argument('-d','--workdir', default='checkpoint', help='Specify the output folder where the DNN weights has to be saved.')
parser.add_argument('--modeldir', default='', help="Specify the directory containing the models. If not specified, it is assumed same as working directory --workdir.")
parser.add_argument('--epochs', default=500, type=int, help='Num. of epochs for training')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for optimizer.')
parser.add_argument('--bs', default=256, type=int, help='Batch size')
parser.add_argument('--nn', default=[256, 128], type=int, nargs='+', help='Number of neurons in each layer')
parser.add_argument('--dropout', default=0.15, type=float, help='Dropout rate to be applied to FC layers')
parser.add_argument('--useBN', action='store_true', help='Apply Batch Normalization.')
parser.add_argument('--method', default='default', help='DNN input generation method.')
parser.add_argument('--excludeBER', action='store_true', help='Used to avoid training on packets that have BER > 0.')
parser.add_argument('--useKeras', action='store_true', help='Used to use tensorflow api rather than keras ones')
parser.add_argument('--useGPU', default='0', help="Select on which GPU to perform compuation.")
parser.add_argument('--valTrainRatio', default=0.15, type=float, help='Set the ratio of validation/training samples.')
parser.add_argument('--valSameTrain', action='store_true', help="Set validation same as training set.")
parser.add_argument('--execTime', action='store_true', help='Perform time measurement during test')
parser.add_argument('--testDropInput', action='store_true', help='Used to test dropout applied on the input')
parser.add_argument('--inFraction', default=1, help='Specify the fraction of the input to pass to the DNN. Ex: 2: first half of the input, 3: first third of the input, etc.')
parser.add_argument('--decimate_max', action='store_true', help='Decimate the input with max pool')
parser.add_argument('--decimate_avg', action='store_true', help='Decimate the input with avg. pool')
parser.add_argument('--onlyReal', action='store_true',help="Train only real model")
parser.add_argument('--onlyImag', action='store_true',help="Train only imag model")
args = parser.parse_args()

# TODO: INCLUDE ARGUMENT TO PERFORM RMS POWER NORMALIZATION ON THE INPUT SIGNAL. THIS HAS TO BE HANDLED BY THE DATA GENERATOR

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import pickle
#from sklearn import preprocessing

if not args.useKeras:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Conv1D, Input, BatchNormalization, AveragePooling1D, MaxPooling1D, Flatten, Dropout, GaussianNoise, Concatenate, Lambda
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, Callback
    from tensorflow import summary
else:
    from keras.models import Model
    from keras.layers import Dense, Conv1D, Input, BatchNormalization, MaxPooling1D, Flatten, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, Callback

#from models import CustomModel

import time
import os
import multiprocessing as mp


os.environ["CUDA_VISIBLE_DEVICES"]=args.useGPU

def userInput():
    yes = {'yes', 'y', 'ye','yo','yep','duh'}
    no = {'no', 'n', '', 'nope'}
    choice = input().lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        print("Please respond with 'yes' or 'no'")

def insertBatch(generator, b, bs, testX_out, testy_out):
    X, y, rms_fact = generator[b]
    if generator.get_method() != 'matlab_maMimo':
        X = [X]
    testX_out[b * bs:(b + 1) * bs] = X[0]
    testy_out[b * bs:(b + 1) * bs] = y

maxproc = mp.cpu_count()
pool = mp.Pool(maxproc)

class changeNoisePower(Callback):

    def __init__(self, avg_sigPow, SNRlevs=[100, 30, 20, 10, 0]):
        self.avg_sigPow = avg_sigPow
        self.SNRlevs = SNRlevs

    def on_train_batch_begin(self, batch, logs=None):
        input_shape = self.model.get_layer(name='AWGN_layer').input.get_shape().as_list()

        SNR = np.random.choice(self.SNRlevs)  # desired SNR for this batch

        noise_pow = self.avg_sigPow / pow(10, SNR / 10)  # noise pow for all signals

        #self.model.layers[1].stddev = np.sqrt(noise_pow)
        self.model.get_layer(name='AWGN_layer').stddev = np.sqrt(noise_pow)/np.sqrt(2)
        #print('updating sttdev in training')
        #print(self.model.layers[1].stddev)

if args.train:
    if os.path.exists(args.workdir) and os.path.isdir(args.workdir):
        print("WARNING: The given output directory exists already. Do you wish to overwrite weight files? [y/N]")
        if not userInput():
            print('Aborting...')
            exit(0)
    else:
        os.mkdir(args.workdir)
elif args.test or args.input_opt:
    if not (os.path.exists(args.workdir) and os.path.isdir(args.workdir)):
        print('Given directory does not exists. Aborting...')
        exit(0)

from massiveMIMO_dataGenerator import loadDataset
# load the dataset and num. of samples based on the experiment/datasource
dataset_train, n_samples, n_train_samples, n_val_samples, dataset_val, simParams = loadDataset(args)

# setup data generators
from massiveMIMO_dataGenerator import DataGenerator
from massiveMIMO_dataGenerator import rms

if args.y == '':    # no validation file is given as input
    if not args.valSameTrain:
        print('Validation separate from Training')
        train_listID = list(range(n_train_samples))
        val_listID = list(range(n_train_samples, n_samples))
        
    else:
        print('WARNING! Validation SAME AS Training!')
        train_listID = list(range(n_samples))
        val_listID = train_listID
else:
    # validation dataset is passed explicitly
    # NOTE: it assumes dataset_val has been previously loaded (depending on datasource)
    print('Using separate input files for training and validation')
    train_listID = list(range(n_samples))
    val_listID = list(range(n_val_samples))

train_generator = dict()
train_generator['real'] = DataGenerator(train_listID, dataset_train, 'real', simParams, model=args.model,
                                        datasource=args.datasource, method=args.method, batch_size=args.bs)
train_generator['imag'] = DataGenerator(train_listID, dataset_train, 'imag', simParams,
                                        datasource=args.datasource, method=args.method, batch_size=args.bs)
valid_generator = dict()
valid_generator['real'] = DataGenerator(val_listID, dataset_val, 'real', simParams,
                                        model=args.model, datasource=args.datasource, method=args.method, batch_size=args.bs)
valid_generator['imag'] = DataGenerator(val_listID, dataset_val, 'imag', simParams,
                                        model=args.model, datasource=args.datasource, method=args.method, batch_size=args.bs)

X, y, rms_fact = train_generator['real'][0]      # obtain the first batch to retrieve size of samples

if args.datasource == 'matlab_maMimo':
    lenSample = X[0].shape[1]
else:
    lenSample = X.shape[1]

# setup model
lr = args.lr
nn = args.nn # neurons

# this is used to test the dropout layer on the input
dropout_test_param = 0.15

dims = ['real', 'imag']
if args.onlyReal:
    dims = ['real']
elif args.onlyImag:
    dims = ['imag']

for d in dims:
    print('Working on *', d, '* model')
    #trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
    if args.datasource == 'matlab_maMimo':
        seq_in = Input(shape=(lenSample,1)) # X[0] = signal
        seq_p = Input(shape=(X[1].shape[1],)) # X[1] = P
    else:
        seq_in = Input(shape=(lenSample,))

    layers = {'dense': [], 'dropout': [], 'batchnorm': [] }
    numDense = len(nn)

    drop_test = Dropout(dropout_test_param, name='drop_test')(seq_in)
    if args.testDropInput:
        next_layer_in = drop_test
    else:
        next_layer_in = seq_in

    if args.method == 'default_SNR' and (not args.test):
        awgn = GaussianNoise(0.5,name='AWGN_layer')(seq_in)
        next_layer_in = awgn

    if args.model == 'FC':

        if args.datasource == 'matlab_maMimo':
            if args.decimate_max:
                decimate = Flatten()(MaxPooling1D()(next_layer_in))
                next_layer_in = Concatenate(axis=1)([decimate, seq_p])
            elif args.decimate_avg:
                decimate = Flatten()(AveragePooling1D()(next_layer_in))
                next_layer_in = Concatenate(axis=1)([decimate, seq_p])
            # TODO aggiungere decimazione pura (senza max o avg pooling, solo scarto)

            else:
                flatten = Flatten()(next_layer_in)
                next_layer_in = Concatenate(axis=1)([flatten, seq_p])


        for i, n in enumerate(nn):
            layers['dense'].append(
                Dense(nn[i], activation='relu', kernel_initializer='glorot_uniform', name='fc_dense'+str(i))(next_layer_in)
            )
            if args.useBN:
                layers['batchnorm'].append(
                    BatchNormalization()(layers['dense'][-1])
                )
                next_layer_in = layers['batchnorm'][-1]
            else:
                next_layer_in = layers['dense'][-1]
            if (i < (numDense - 1)) and (args.dropout != 0.0):  # if not the last layer, add dropout
                layers['dropout'].append(
                    Dropout(args.dropout, name='drop'+str(i))(next_layer_in)
                )
                next_layer_in = layers['dropout'][-1]
        encoder = Dense(simParams['nSubCarr'], activation='linear', kernel_initializer='glorot_uniform', name='fc_regressor')(next_layer_in)



        if args.datasource == 'matlab_maMimo':
            CSI_predictor = Model([seq_in,seq_p], encoder)
        else:
            CSI_predictor = Model(seq_in, encoder)

    elif args.model =='CONV1D':

        conv1 = BatchNormalization()(Conv1D(128, 7, padding='same', name='cnn1d_1', activation='relu')(next_layer_in))
        maxpool1 = AveragePooling1D()(conv1)
        """
        conv2 = BatchNormalization()(Conv1D(64, 7, padding='same', name='cnn1d_2', activation='relu')(maxpool1))
        maxpool2 = AveragePooling1D()(conv2)
       
        conv3 = BatchNormalization()(Conv1D(64, 5, padding='same', name='cnn1d_3', activation='relu')(maxpool2))
        maxpool3 = MaxPooling1D()(conv3)
        """
        flatten = Flatten()(maxpool1)
        # TODO INSERT CONVOLUTIONAL AUTOENCODER HERE

        next_layer_in = Concatenate(axis=1)([flatten, seq_p])

        for i, n in enumerate(nn):
            layers['dense'].append(
                Dense(nn[i], activation='relu', kernel_initializer='glorot_uniform', name='fc_dense'+str(i))(next_layer_in)
            )
            if args.useBN:
                layers['batchnorm'].append(
                    BatchNormalization()(layers['dense'][-1])
                )
                next_layer_in = layers['batchnorm'][-1]
            else:
                next_layer_in = layers['dense'][-1]
            if i < (numDense - 1):  # if not the last layer, add dropout
                layers['dropout'].append(
                    Dropout(args.dropout, name='drop'+str(i))(next_layer_in)
                )
                next_layer_in = layers['dropout'][-1]
        encoder = Dense(n_out, activation='linear', kernel_initializer='glorot_uniform', name='fc_regressor')(next_layer_in)

        CSI_predictor = Model([seq_in,seq_p], encoder)

    CSI_predictor.summary()

    opt = Adam(lr=lr)
    CSI_predictor.compile(optimizer=opt, loss='mse')

    # setup callbacks
    if args.modeldir == '':
        model_filepath = os.path.join(args.workdir, d + "_weights-improvement.hdf5")
    else:
        model_filepath = os.path.join(args.modeldir, d + "_weights-improvement.hdf5")

    if args.train:
        # setup callbacks
        earlystop = EarlyStopping(monitor='val_loss',mode='min', patience=25, verbose=1,restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=args.lr*0.01, verbose=True )

        if args.datasource == 'matlab_WiFi_SISO' or args.datasource == 'matlab_SpiNN_WiFi' or args.datasource == 'Carlos-gnuradio':
             #cb_list = [earlystop]      # TODO for SISO it triggers ReduceLROnPlateau too soon!! Why??????
             cb_list = [earlystop,reduce_lr]
        else:
            # cb_list = [earlystop]
            cb_list = [earlystop,reduce_lr]

        if args.method == 'default_SNR':
            # TODO: in case we use power normalization, noise application should be performed before power normalization
            # compute average sig. power based on first minibatch - TODO: (would be better to be performed over all dataset)
            X, y, rms_fact = train_generator['real'][0]

            if args.datasource == 'matlab_maMimo':
                sigPows = (np.apply_along_axis(rms, 1, X[0])) ** 2
                avg_sigPow = np.mean(sigPows)
                applyGaussNoise = changeNoisePower(avg_sigPow=avg_sigPow,SNRlevs=[30, 20, 10, 0, -10, -20])  # balanceNoise -> current, moreNoise = [120, 30, 10, 0, -10, -20]
            elif args.datasource == 'matlab_WiFi_SISO' or args.datasource == 'matlab_SpiNN_WiFi' or args.datasource == 'Carlos-gnuradio':
                sigPows = (np.apply_along_axis(rms, 1, X)) ** 2
                avg_sigPow = np.mean(sigPows)
                applyGaussNoise = changeNoisePower(avg_sigPow=avg_sigPow)
            cb_list.append(applyGaussNoise)


        # start fitting
        pred_history = CSI_predictor.fit(x=train_generator[d],
                                validation_data=valid_generator[d],
                                epochs=args.epochs,
                                callbacks=cb_list
                               )

        # after training, save the best model configuration (restored by earlystop callback)
        CSI_predictor.save_weights(model_filepath)

        plt.semilogy(pred_history.history['loss'])
        plt.semilogy(pred_history.history['val_loss'])
        plt.title('model loss for CSI mapping')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper left')
        plt.savefig(os.path.join(args.workdir, d + '_loss_prediction'))
        plt.clf()

    elif args.test:

        # load model and test input data
        # load the model from best checkpoint
        CSI_predictor.load_weights(model_filepath)

        # restore original samples order to be read in matlab test script
        valid_generator[d].reorder_indexes()
        # set the batchsize as num. of nTx * nRx
        valid_generator[d].set_batchsize(simParams['nTX']*simParams['nRX'])

        if not args.testDropInput:
            # evaluate the model performance on test data
            CSI_predictor.evaluate(x=valid_generator[d], verbose=1)

            # generate prediction output
            csi_out = CSI_predictor.predict(x=valid_generator[d])
            # export all batches for matlab test
            bs = valid_generator[d].get_batchsize()

            X, y, rms_fact = valid_generator[d][0]

            if valid_generator[d].get_method() != 'matlab_maMimo':
                X = [X]

            testX_out = np.empty((bs*len(valid_generator[d]), X[0].shape[1]))
            testy_out = np.empty((bs*len(valid_generator[d]), y.shape[1]))



            func_args = [[valid_generator[d], b, bs, testX_out, testy_out] for b in range(len(valid_generator[d]))]
            out_data = pool.starmap(insertBatch, func_args)

            """
            testX_out = None
            testy_out = None

            for b in range(len(valid_generator[d])):
                X, y = valid_generator[d][b]
                if b == 0:
                    testX_out = X[0]
                    testy_out = y
                else:
                    testX_out = np.concatenate((testX_out, X[0]), axis=0)
                    testy_out = np.concatenate((testy_out, y), axis=0)
            """
        else:

            testX_out = None
            testy_out = None
            csi_out = None

            for b in range(len(valid_generator[d])):
                X, y = valid_generator[d][b]
                # generate a mask to simulate dropout on the input
                dropout_mask = np.random.binomial(1, 1 - dropout_test_param, X.shape)
                X *= dropout_mask
                # generate prediction output
                y_dnn = CSI_predictor.predict(x=X)
                # store outcome in output data
                if b == 0:
                    testX_out = X
                    testy_out = y
                    csi_out = y_dnn
                else:
                    testX_out = np.concatenate((testX_out, X), axis=0)
                    testy_out = np.concatenate((testy_out, y), axis=0)
                    csi_out = np.concatenate((csi_out, y_dnn), axis=0)


        # save different .mat file for every packet, instead of a single one (savemat has size limits)
        batch_size = valid_generator[d].get_batchsize() # during test, this is equal to simParams['nTX']*simParams['nRX']
        pkt_count = 1
        for h in range(0,len(valid_generator[d])*batch_size,batch_size):
            mat_out = {'all_pkts_csi_nn_out': dict(x=testX_out[h:h+batch_size],
                                                   y=csi_out[h:h+batch_size],
                                                   true_y=testy_out[h:h+batch_size])}
            file_ID = int(h / batch_size)+1
            savemat(os.path.join(args.workdir, 'test_csi_predictions_' + d +'_'+str(file_ID)+'.mat'), mat_out,  do_compression=True)

        CSI_predictor.save(os.path.join(args.workdir, d + "_keras_model")) # add .h5 extension for keras model format

        #valid_generator[d].on_epoch_end() # trigger shuffling of indexes
        X,y,rms_fact = valid_generator[d][0]    # load some data from the dataset

        if args.datasource != 'matlab_maMimo':
            X = [X];

        if X[0].shape[0] >= 20:
            n_pics = 20
        else:
            batch = 0
            n_pics = X[0].shape[0]
            while n_pics < 20:
                batch += 1
                X_temp, y_temp, rms_fact_temp = valid_generator[d][batch]
                X[0] = np.concatenate((X[0], X_temp), axis=0)
                y = np.concatenate((y, y_temp), axis=0)
                n_pics += X_temp.shape[0]


        for i in range(n_pics):
            if args.datasource == 'matlab_maMimo':
                plt.plot(CSI_predictor.predict([X[0][i:i+1],X[1][i:i+1]])[0, :])
            else:
                plt.plot(CSI_predictor.predict(X[0][i:i + 1])[0, :])
            plt.plot(y[i:i+1][0, :])
            plt.savefig(os.path.join(args.workdir, str(i)+'_'+d+'_bluePred.png'))
            plt.clf()

        if args.execTime:
            print("******** Check timings!! ********")
            """
            iters = 1000
            N_ats = [32, 64, 128, 256, 512]
            # check timing for N_ats antennas prediction, run for 1000 times and average the time
            import statistics

            for a in N_ats:
                timings = []
                for n in range(iters):
                    start = time.process_time()
                    CSI_predictor.predict(X[0:a], batch_size=a)
                    end = time.process_time()
                    timings.append((end - start))
                print('N.channels: ' + str(a))
                print("Time forward sample (average exec.): {} ".format(statistics.mean(timings)))
                print('--------------')
            """

            iters = 10
            N_channels = dataset_train['simParams']['nTX'] * dataset_train['simParams']['nRX']
            print("Testing", N_channels, "per batch")

            for n in range(iters):
                # Set up logging.
                logdir = os.path.join(args.workdir, "logs_inf/inference_" + d + "_i" + str(n))
                writer = summary.create_file_writer(logdir)
                summary.trace_on(graph=True, profiler=True)
                CSI_predictor.predict([X[0][0:N_channels],X[1][0:N_channels]], batch_size=N_channels)
                with writer.as_default():
                    summary.trace_export(
                        name="inference_trace",
                        step=0,
                        profiler_outdir=logdir)
    elif args.input_opt:
        print("TODO")

if args.train:
    #save also hyper parameters
    hpp = {'args': args, 'optimizer': opt, 'lr': lr,  }
    pickle.dump(hpp, open(os.path.join(args.workdir, "train_hparam.pkl"), 'wb'))






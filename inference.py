import os
import sys
import numpy as np
from tensorflow import keras

class CSIPredictor:

    def __init__(self, model_path, experiment='RICE_RENEW', verbose=False):
        self.path = model_path
        self.experiment = experiment
        self.verbose = verbose
        self.model_real, self.model_imag = self.load_model()

    def load_model(self):
        model_real = keras.models.load_model(os.path.join(self.path, 'real_keras_model'))
        model_imag = keras.models.load_model(os.path.join(self.path, 'imag_keras_model'))
        if self.verbose:
            print('------- Real Model Summary -------')
            model_real.summary()
            print('------- Imag Model Summary -------')
            model_imag.summary()
        return model_real, model_imag

    def inference(self, input_batch: np.ndarray):

        X = self.preprocess_data(input_batch)
        bs = X.shape[0]   # assumes num. of samples in the first dimension
        # perform prediction on real and imaginary models
        output_real = self.model_real.predict(X.real, batch_size=bs)
        output_imag = self.model_imag.predict(X.imag, batch_size=bs)
        output_cmplx = output_real + 1j * output_imag   # create complex data
        return self.postprocess_data(output_cmplx)


    def preprocess_data(self, input_batch):
        """
        In this function we perform data preparation specific to each experiment
        """
        if self.experiment == 'RICE_RENEW':
            # we expect input_batch to be an array with type np.complex128
            if input_batch.dtype != np.complex128:
                print('[CSIPredictor] ERROR: Input batch must be of type np.complex128')
                sys.exit(-1)
            prep_data = input_batch

        return prep_data

    def postprocess_data(self, output_batch):
        """
        In this function we perform post-processing of DNN output specific to each experiment
        """
        if self.experiment == 'RICE_RENEW':
            # we need to reinsert null values and revert the FFT Shift operation,
            # since in RICE-RENEW code they don't do that
            # IMPORTANT!!! Assuming FFT size is always 64
            if output_batch.shape[1] == 52:     # 52 are non-zero subcarriers (pilots+data)
                # concatenate zeroes to all batch output samples, according to zero-valued subcarriers arrangement
                tmp = np.concatenate((np.zeros((output_batch.shape[0], 6)),
                                       output_batch[:, 0:26],
                                       np.zeros((output_batch.shape[0], 1)),
                                       output_batch[:, 26:],
                                       np.zeros((output_batch.shape[0], 5))), axis=1)
                postp_data = np.fft.ifftshift(tmp, axes=1)
            else:
                print('[CSIPredictor] ERROR: Output samples must have size 52 (assuming FFTLen = 64).')
                sys.exit(-1)

        return postp_data
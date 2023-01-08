from tensorflow.keras.layers import Input, TimeDistributed, BatchNormalization, Conv2D, ConvLSTM2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from utils.PlotterTensorboard import PlotterTensorboard

from tensorflow.keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

from dataio.datareader import datareader

from config import LSTM_CFG

class TDCNNv1:
    '''
        This class implements a Time Distributed Convolutional Neural Networ. 

        The implementation has been adapted to work with Earth Observation data.
    '''
    def __init__(self, shape):
        '''
            The constructure takes care of creating the TDCNN object and contains all the
            functions used to create and train/test the TDCNN model. The settings for the
            model are saved into a config file.

            Inputs:
                - shape: input shape for the TDCNN. Must be a 4-D tuple (T,W,H,B) with T
                         size of temporal series, W image width, H image height and B number
                         of image channels (bands).
        '''
        self.shape        = shape[1:]
        self.t_len        = shape[0]
        self.depth        = LSTM_CFG['FILTERS']
        self.kernels      = LSTM_CFG['KERNELS']
        self.activations  = LSTM_CFG['ACTIVATIONS']
        self.loss         = LSTM_CFG['LOSS']
        self.lr           = LSTM_CFG['LEARNING_RATE']
        
        self.bs           = LSTM_CFG['BATCH_SIZE']
        self.epochs       = LSTM_CFG['EPOCHS']
        self.es_rounds    = LSTM_CFG['EARLY_STOPPING_ROUNDS']
        
        # Build and compile the model
        self.model = self.__build()

    def __build(self):
        '''
            This prive method builds the TDCNN model.

            Outputs:
                - model: built and compiled TDCNN model.
        '''
        x_in = Input(shape = (self.t_len-1, None, None, self.shape[-1]))
        x = x_in
        
        # Based on the desired depth, the following chunk of code will append
        # ConvLSTM layers to the model
        for i, filt in enumerate(self.depth):
		    
            # The return sequences parameters of the last layer only, is set to
            # False.

            return_sequences = True
            if i == (len(self.depth) - 1): return_sequences = False
            
            x = TimeDistributed(Conv2D(
                    filters          = filt,
                    kernel_size      = self.kernels[i],
                    padding          = 'same',
                    activation       = self.activations[i]))(x)
            x = TimeDistributed(BatchNormalization())(x)
        
        x = ConvLSTM2D(
                filters          = filt,
                kernel_size      = self.kernels[-1],
                padding          = 'same',
                activation       = self.activations[i],
                return_sequences = False)(x)
        
        # The final layer is a Conv2D layer
        x = Conv2D(self.shape[-1], kernel_size = (3,3), activation=self.activations[-1], padding='same')(x)
        # Create the model
        model = Model(inputs = x_in, outputs=x, name = 'TDCNN')
        # Compile the model with optimizer and loss
        model.compile(optimizer = Adam(self.lr), loss = self.loss) 

        return model
    
    def train(self, train_set, val_set, normalize=True):
        '''
            This public method will take care of the training process.
            
            Inputs:
                - train_set: python dictionary -> keys: geo locations, values: image time sequences paths
                - val_set: as above, but for validation
                - normalize: boolean flag used to specify if the images need to be normalized or not
            Outputs:
                - history: training history
            
            ----------------------------------------------------------------------------------------------
            Please note that a TensorBoard callbak is associated with this method. If you want to monitor
            the training you can lunch the following command in the terminal:
            
            $ tensorboard --logdir tmp/TDCNNv1

            Please note that for each training process a specific folder will be created using the date and
            time of the execution. In this way you can monitor several experiments. The callback will show
            the training curves as well as some images predictions on validation set.
        '''

        # Training and Validation data loader
        train_gen = datareader.generator(train_set, 
                                         self.bs,
                                         self.t_len,
                                         self.shape,
                                         normalize=normalize)
        val_gen   = datareader.generator(val_set,
                                         self.bs,
                                         self.t_len,
                                         self.shape,
                                         normalize=normalize)
        val_gen_2 = datareader.generatorv2(val_set,
                                           self.bs*10,
                                           self.t_len,
                                           self.shape,
                                           normalize=normalize)
        # Callbacks
        tb_path = os.path.join('tmp', ' TDCNNv1', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        tb = TensorBoard(log_dir=tb_path, histogram_freq=1)
        es = EarlyStopping(monitor='val_loss', 
                           patience=self.es_rounds,
                           mode='auto',
                           verbose=0,
                           baseline=None)
        pl = PlotterTensorboard(model = self.model, generator = val_gen_2, log_path = tb_path)       
    
        # Training
        hist = self.model.fit(
                                  train_gen,
            steps_per_epoch     = len(train_set.values())//self.bs,
            epochs              = self.epochs,
            validation_data     = val_gen,
            validation_steps    = len(val_set.values())//self.bs,
            callbacks           = [es, tb, pl]
        )

        return hist
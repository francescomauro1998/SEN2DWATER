from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

from dataio.datareader import datareader

from config import LSTM_CFG

class BLSTMv1:
    '''
        This class implements a Bidirectional Convolutional Long Short-Term Memory. The implementation
        is based on the Keras Next-Frame Video Prediction model presented here:
        https://keras.io/examples/vision/conv_lstm/

        The implementation has been adapted to work with Earth Observation data.
    '''
    def __init__(self, shape):
        '''
            The constructure takes care of creating the BLSTM object and contains all the
            functions used to create and train/test the BLSTM model. The settings for the
            model are saved into a config file.

            Inputs:
                - shape: input shape for the BLSTM. Must be a 4-D tuple (T,W,H,B) with T
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
            This prive method builds the BLSTM model.

            Outputs:
                - model: built and compiled BLSTM model.
        '''
        x_in = Input(shape = (self.t_len-1, None, None, self.shape[-1]))
        x = x_in
        
        # Based on the desired depth, the following chunk of code will append
        # BLSTM layers to the model
        for i, filt in enumerate(self.depth):
		    
            # The return sequences parameters of the last layer only, is set to
            # False.

            return_sequences = True
            if i == (len(self.depth) - 1): return_sequences = False
            
            x = Bidirectional(
                ConvLSTM2D(
                    filters          = filt,
                    kernel_size      = self.kernels[i],
                    padding          = 'same',
                    activation       = self.activations[i],
                    return_sequences = return_sequences))(x)
            x = BatchNormalization()(x)
        
        # The final layer is a Conv2D layer
        x = Conv2D(self.shape[-1], kernel_size = (3,3), activation=self.activations[-1], padding='same')(x)
        # Create the model
        model = Model(inputs = x_in, outputs=x, name = 'BLSTM')
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
            
            $ tensorboard --logdir tmp/BLSTM

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
                                           self.bs,
                                           self.t_len,
                                           self.shape,
                                           normalize=normalize)
        # Callbacks
        tb_path = os.path.join('tmp', 'BLSTMv1', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
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


class PlotterTensorboard(Callback):
    '''
        This class relates to the TensorBoard callbacks specified in train function above.
    '''

    def __init__(self, model, generator, log_path):
        self.generator  = generator
        self.model      = model
        self.log_path   = log_path
        self.writer     = tf.summary.create_file_writer(log_path)

    def on_epoch_end(self, epoch, logs={}):
        x_in, y_in = self.generator
        y_pr = self.model.predict(x_in, verbose=0)

        # Tensorboard visualization
        with self.writer.as_default():
            tf.summary.image(name='Ground Truth', data=(y_in*255).astype(np.uint8), step=epoch, max_outputs=3)
            tf.summary.image(name='Prediction',   data=(y_pr*255).astype(np.uint8), step=epoch, max_outputs=3)
        
        # Save results
        gt_path = os.path.join(self.log_path, 'res', 'gt', 'epoch-{}'.format(epoch))
        pr_path = os.path.join(self.log_path, 'res', 'pr', 'epoch-{}'.format(epoch))

        os.makedirs(gt_path, exist_ok = True)
        os.makedirs(pr_path, exist_ok = True)
        for i in range(y_in.shape[0]):
            plt.imsave(os.path.join(gt_path,'gt-{}.png'.format(i)), (y_in[i,...,0]*255).astype(np.uint8))
            plt.imsave(os.path.join(pr_path,'pt-{}.png'.format(i)), (y_pr[i,...,0]*255).astype(np.uint8))
       

from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D
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

from .config import LSTM_CFG

class LSTMv1:

    def __init__(self, shape):
        self.shape        = shape
        self.depth        = LSTM_CFG['FILTERS']
        self.kernels      = LSTM_CFG['KERNELS']
        self.activations  = LSTM_CFG['ACTIVATIONS']
        self.loss         = LSTM_CFG['LOSS']
        self.lr           = LSTM_CFG['LEARNING_RATE']
        
        self.bs           = LSTM_CFG['BATCH_SIZE']
        self.epochs       = LSTM_CFG['EPOCHS']
        self.es_rounds    = LSTM_CFG['EARLY_STOPPING_ROUNDS']
        
        self.model = self.__build()

    def __build(self):
        x_in = Input(shape = self.shape)
        x = x_in

        for i, filt in enumerate(self.depth):
		    
            return_sequences = True
            if i == (len(self.depth) - 1): return_sequences = False

            x = ConvLSTM2D(
                    filters          = filt,
                    kernel_size      = self.kernels[i],
                    padding          = 'same',
                    activation       = self.activations[i],
                    return_sequences = return_sequences)(x)
            x = BatchNormalization()(x)

        x = Conv2D(self.shape[-1], kernel_size = (3,3), activation='sigmoid', padding='same')(x)

        model = Model(inputs = x_in, outputs=x, name = 'LSTMv1')
        model.compile(optimizer = Adam(self.lr), loss = self.loss) 

        return model
    
    def train(self, train_set, val_set, normalize=True):
        # Training and Validation data loader
        train_gen = datareader.generator(train_set, 
                                         self.bs,
                                         self.shape,
                                         normalize=normalize)
        val_gen   = datareader.generator(val_set,
                                         self.bs,
                                         self.shape,
                                         normalize=normalize)
        val_gen_2 = datareader.generatorv2(val_set,
                                           self.bs,
                                           self.shape,
                                           normalize=normalize)
        # Callbacks
        tb_path = os.path.join('tmp', 'LSTMv1', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
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
            steps_per_epoch     = len(train_set[0])//self.bs,
            epochs              = self.epochs,
            validation_data     = val_gen,
            validation_steps    = len(val_set[0])//self.bs,
            callbacks           = [es, tb, pl]
        )

        return hist


class PlotterTensorboard(Callback):
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
            tf.summary.image(name='Ground Truth', data=y_in, step=epoch, max_outputs=3)
            tf.summary.image(name='Prediction',   data=y_pr, step=epoch, max_outputs=3)
        
        # Save results
        #gt_path = os.path.join(self.log_path, 'res', 'gt', 'epoch-{}'.format(epoch))
        #pr_path = os.path.join(self.log_path, 'res', 'pr', 'epoch-{}'.format(epoch))

        #os.makedirs(gt_path, exist_ok = True)
        #os.makedirs(pr_path, exist_ok = True)
        #for i in range(x_in.shape[0]):
        #    plt.imsave(os.path.join(gt_path,'gt-{}.png'.format(i)), (x_in[i,...]*255).astype(np.uint8))
        #    plt.imsave(os.path.join(pr_path,'pt-{}.png'.format(i)), (y_pr[i,...]*255).astype(np.uint8))
       

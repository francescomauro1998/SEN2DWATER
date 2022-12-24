from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.model import Model

from .config import config

class LSTMv1:

	def __init__(self, shape):
        
        self.shape        = shape
        self.depth        = LSTM_CFG['FILTERS']
        self.kernels      = LSTM_CFG['KERNEL']
        self.activations  = LSTM_CFG['ACTIVATIONS']
        self.loss         = LSTM_CFG['LOSS']
        self.lr           = LSTM_CFG['LEARNING_RATE']

		self.model = self.__build()
        
	def __build(self):
       	x_in = Input(shape = self.shape)
		x = x_in

	    for i, filt in enumerate(self.depth):
		    
            if i < len(self.depth): return_sequence = True
            else: return_sequence = False

             x = ConvLSTM2D(
                    filters         = filt,
                    kernel_size     = self.kernels[i],
                    padding         = 'same',
                    activation      = self.activations[i],
                    return_sequence = return_sequence)(x)
            x = BatchNormalization()(x)

        x = Conv2D(self.shape[-1], kernel_size = (3,3), activation='sigmoid', padding_same)(x)

        model = Model(inputs = x_in, outputs=x, name = 'LSTMv1')
        model.compile(optimizer = Adam(self.lr), loss = self.loss) 

        return model
    
    def train(self, train_set, val_set):







                

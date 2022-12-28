import os

# This represents the root path for the dataset, arrange it as it suits you
dataset_root_path = os.path.join('datasets', 'S2-water')
SPLIT_FACTOR      = 0.5

# Image settings to feed LSTM
T_LEN           = 5
PATCH_WIDTH     = 64
PATCH_HEIGHT    = 64
BANDS           = 1

# This dictionary contains all the settins for the LSTM
LSTM_CFG = {
                'FILTERS':                  [64, 32, 16],
                'KERNELS':                  [3, 2, 1],
                'ACTIVATIONS':              ['tanh', 'tanh', 'tanh'],
                'LOSS':                     'huber',
                'LEARNING_RATE':            0.0002,
                'BATCH_SIZE':               1,
                'EPOCHS':                   50,
                'EARLY_STOPPING_ROUNDS':    10
        }

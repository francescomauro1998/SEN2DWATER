import os

# This represents the root path for the dataset, arrange it as it suits you
dataset_root_path = os.path.join('datasets', 'DATASET-1-v2')
SPLIT_FACTOR      = 0.2

# Image settings to feed LSTM
T_LEN           = 7
PATCH_WIDTH     = 64
PATCH_HEIGHT    = 64
BANDS           = 1

# This dictionary contains all the settins for the LSTM
LSTM_CFG = {
                'FILTERS':                  [32, 32, 32, 32],
                'KERNELS':                  [3, 3, 3, 3],
                'ACTIVATIONS':              ['tanh', 'tanh', 'tanh', 'tanh'],
                'LOSS':                     'huber',
                'LEARNING_RATE':            0.00002,
                'BATCH_SIZE':               10,
                'EPOCHS':                   50,
                'EARLY_STOPPING_ROUNDS':    10
        }

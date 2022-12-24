LSTM_CFG = {
                'FILTERS':                  [64, 32, 16],
                'KERNELS':                  [3, 2, 1],
                'ACTIVATIONS':              ['tanh', 'tanh', 'tanh'],
                'LOSS':                     'huber',
                'LEARNING_RATE':            0.0002,
                'BATCH_SIZE':               16,
                'EPOCHS':                   50,
                'EARLY_STOPPING_ROUNDS':    10
        }

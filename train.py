from dataio.datahandler import datahandler
from dataio.datareader import datareader
from models.LSTMv1 import LSTMv1

from utils.plot_dataset import plot_series 

from config import *


# Loading the dataset
dh = datahandler(dataset_root_path)
keys = list(dh.paths.keys())
t_len = len(dh.paths[keys[0]])

print('{:=^100}'.format(' Loading the dataset '))
print('\t -{:<50s} {}'.format('Number of GeoLocation', len(keys)))
print('\t -{:<50s} {}'.format('Number of Images per GeoLocation', t_len))

# Plotting sample from the dataset
plot_series(dataset_root_path, (300,300, 16), 3, t_len = t_len, n_samples = 2)

# Building the model
lstmv1 = LSTMv1(shape=(T_LEN, PATCH_WIDTH, PATCH_HEIGHT, BANDS))

print('{:=^100}'.format(' Building the Model '))
print(lstmv1.model.summary())

from dataio.datareader import datareader
from dataio.datahandler import datahandler
from models.LSTMv1 import LSTMv1

from config import *


# Loading the dataset
dh = datahandler(dataset_root_path)
keys = list(dh.paths.keys())

print('{:=^100}'.format(' Loading the dataset '))
print('\t {:<50s} {}'.format('Number of GeoLocation', len(keys)))
print('\t {:<50s} {}'.format('Number of Images per GeoLocation', len(dh.paths[keys[0]])))


# Building the model
lstmv1 = LSTMv1(shape=(38, 64, 64, 1))

print('{:=^100}'.format(' Building the Model '))
print(lstmv1.model.summary())

from dataio.datahandler import datahandler
from dataio.datareader import datareader
from models.LSTMv1 import LSTMv1

import matplotlib.pyplot as plt

from config import *

# Loading the dataset
dh       = datahandler(dataset_root_path)
keys     = list(dh.paths.keys())
t_len    = len(dh.paths[keys[0]])

print('{:=^100}'.format(' Loading the dataset '))
print('\t -{:<50s} {}'.format('Number of GeoLocation', len(keys)))
print('\t -{:<50s} {}'.format('Number of Images per GeoLocation', t_len))



b_in, b_ou = next(iter(datareader.generator(dh.paths, 2, 6, (300,300,13), True)))

print(b_in.shape)
print(b_ou.shape)

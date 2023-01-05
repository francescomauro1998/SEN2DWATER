from dataio.datahandler import datahandler
from dataio.datareader import datareader
from models.LSTMv1 import LSTMv1
from models.BLSTMv1 import BLSTMv1
from models.TDCNNv1 import TDCNNv1

from utils.plot_dataset import plot_series 

from config import *


#======================================= Loading the dataset ========================================
dh       = datahandler(dataset_root_path)
keys     = list(dh.paths.keys())
t_len    = len(dh.paths[keys[0]])

print('{:=^100}'.format(' Loading the dataset '))
print('\t -{:<50s} {}'.format('Number of GeoLocation', len(keys)))
print('\t -{:<50s} {}'.format('Number of Images per GeoLocation', t_len))

#========================================== Split dataset ===========================================
train_set, val_set = dh.split(SPLIT_FACTOR)
print('{:=^100}'.format(' Splitting the dataset '))
print('\t -{:<50s} {}'.format('Number of GeoLocation (training)', len(train_set.keys())))
print('\t -{:<50s} {}'.format('Number of GeoLocation (validation)', len(val_set.keys())))

#========================================== Build model =============================================
lstmv1 = BLSTMv1(shape=(T_LEN, PATCH_WIDTH, PATCH_HEIGHT, BANDS))

print('{:=^100}'.format(' Building the Model '))
print(lstmv1.model.summary())

#========================================== Train model =============================================
print('{:=^100}'.format(' Training the Model '))

history = lstmv1.train(train_set, val_set, normalize = True)


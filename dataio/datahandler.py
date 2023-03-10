import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import numpy as np
import glob
import os

class datahandler:
    '''
        Data Handler, it handle the dataset
    '''

    def __init__(self, dataset_root):
        '''
            It creates a Data Handler object.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            The dataset must be organized in this format:

            RootFolder
            │   README.md
            │   requirements.txt    
            │
            └───dataset_root
                └───zone_1
                    └───t0.tif
                    └───t1.tif
                    └─── ...
                    └───t30.tif
                         
            The following image format are supported:
                - .png
                - .jpg
                - .jpeg
                - .tif
                - .npy
            please adapt your format accordingly. 

            Images should contain all the bands in a single file.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            Input
                - root: root path of the dataset
        '''

        self.dataset_root = dataset_root
        self.paths = self.__load_paths()
    
    def __load_paths(self, verbose=False):
        '''
            Loads the images path

            Output:
                - paths: python dictionary, keys==locations, values==paths of timeseries
        '''

        geo_locations = glob.glob(os.path.join(self.dataset_root, '*'))

        paths = {}
        for i, c in tqdm(enumerate(geo_locations), disable=not(verbose), colour='blue'):
            imgs_c = glob.glob(os.path.join(c, '*'))
            imgs_c.sort()
            paths[c.split(os.sep)[-1]] = imgs_c

        return paths
    
    def split(self, split_factor = 0.2):
        '''
            Splits the dataset in training and validation sets.

            Inputs:
                - split_factor: float number [0,1] used to split the dataset
            Outputs:
                - (train_set, val_set): tuple containing training set and validation set

        '''
        
        l = len(list(self.paths.keys()))

        vl = int(l*split_factor)
        tl = l - vl

        train_set = {k:self.paths[k] for k in list(self.paths.keys())[:tl]}
        val_set   = {k:self.paths[k] for k in list(self.paths.keys())[tl:]}        
        return train_set, val_set

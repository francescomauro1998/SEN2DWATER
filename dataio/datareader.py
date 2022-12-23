from .normalizer import normalizer
import matplotlib.pyplot as plt
from tqmd.auto import tqdm
import numpy as np
import rasterio
import random
import cv2

class datareader:
    '''
        Class containing static methods for reading images
    '''

    @staticmethod
    def load(path):
        '''
            Load an image and its metadata given its path.
            
            The following image format are supported:
                - .png
                - .jpg
                - .jpeg
                - .tif
                - .npy
            please adapt your format accordingly. 
            
            Inputs:
                - path: position of the image, if None the function will ask for the image path using a menu
                - info (optional): allows to print process informations
            Outputs:
                - data: WxHxB image, with W width, H height and B bands
                - metadata: dictionary containing image metadata
        '''
        
        MPL_FORMAT = ['.png', '.jpg', '.jpeg']
        RIO_FORMAT = ['.tif', '.tiff']
        NP_FORMAT  = ['.npy']
        
        if any(frmt in path for frmt in RIO_FORMAT):
            with rasterio.open(path) as src:
                data     = src.read()
                metadata = src.profile
            data = np.moveaxis(data, 0, -1)
            
        elif any(frmt in path for frmt in MPL_FORMAT):
            data = plt.imread(path)
            metadata = None

        elif any(frmt in path for frmt in NP_FORMAT):
            data     = np.load(path)
            metadata = None
            
        else:
            data     = None
            metadata = None
            print('!!! File can not be opened, format not supported !!!')
            
        return data, metadata

    @staticmethod
    def load_samples(dataset, n_samples, img_shape, normalize = True):
        '''
            Load N samples from the dataset

            Inputs:
                - dataset: python dictionary, keys==locations, values==paths of timeseries
                - n_samples: number of samples to be loaded
                - img_shape: shape of the images to be loaded
                - normalies: if true normalize the images
            Output:
                - imgs: loaded images, 1st dim. samples, 2nd dim. time, 3rd-4th dims. space, 5th dim. bands

        '''

        imgs = np.zeros((n_samples, len(dataset[0]), img_shape[0], img_shape[1], img_shape[2]))

        for n in tqdm(range(n_samples), colour='black'):
            imgs_path = dataset[n]
            for t in imgs_path:
                img = self.load(imgs_path[t])

                if normalize != None: img = normalizer.minmax_scaler(img)
                if img_shape != img.shape: img = cv2.resize(img, img_shape[:2])

                imgs[n, t, ...] = img


        return imgs
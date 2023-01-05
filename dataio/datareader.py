from .normalizer import normalizer
from .spectral_indices import spectral_indices
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
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
    def save(image, path, meta):
        '''
            Save an image and its metadata given its path
            Inputs:
                - image: the image to be saved
                - path: position of the image
                - meta: metadata for the image to be saved
        '''

        RASTERIO_EXTENSIONS   = ['.tif', '.tiff']
        MATPLOTLIB_EXTENSIONS = ['.png', '.jpg', 'jpeg']

        if any(frmt in path for frmt in RASTERIO_EXTENSIONS):

            if meta!=None:
                meta.update({'driver':'GTiff',
                            'width':image.shape[0],
                            'height':image.shape[1],
                            'count':image.shape[2],
                            'dtype':'float64'})

            with rasterio.open(fp=path, mode='w',**meta) as dst:
                for count in range(image.shape[2]):
                    dst.write(image[:,:,count], count+1)

        elif any(frmt in path for frmt in MATPLOTLIB_EXTENSIONS):
            plt.imsave(path, image)

        else:
            print('[!] File cannot be saved, format not supported!')

    @staticmethod
    def generator(dataset, batch_size, t_len, img_shape, normalize=True):
        '''
            TO-DO
        '''

        x_in = np.zeros((batch_size, t_len-1, img_shape[0], img_shape[1], 1))
        x_ou = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        dataset = list(dataset.values())
        random.shuffle(dataset)

        counter = 0

        while True:

            if counter > (len(dataset) - batch_size): 
                counter = 0
                random.shuffle(dataset)

            for b in range(batch_size):
                paths = dataset[counter]
                counter += 1

                for i in range(t_len):
                    img, _ = datareader.load(paths[i])
                     
                    if normalize != None: img = normalizer.max_scaler(img, 10000)
                    if img_shape != img.shape: img = cv2.resize(img, img_shape[:2])
                     
                    ndwi = spectral_indices.normalized_difference(img, [2,7])
                    
                    if i < (t_len - 1):  x_in[b, i, :, :, 0] = ndwi 
                    else:                x_ou[b,    :, :, 0] = ndwi
            
            yield x_in, x_ou
    
    @staticmethod
    def generatorv2(dataset, batch_size, t_len, img_shape, normalize=True):
        '''
            TO-DO
        '''
        x_in = np.zeros((batch_size, t_len-1, img_shape[0], img_shape[1], 1))
        x_ou = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        dataset = list(dataset.values())
        rdn = (len(dataset)//batch_size)-batch_size
        
        for b in range(batch_size):
            paths = dataset[b]

            for i in range(t_len):
                img, _ = datareader.load(paths[i])

                if normalize != None: img = normalizer.max_scaler(img, 10000)
                if img_shape != img.shape: img = cv2.resize(img, img_shape[:2])

                ndwi = spectral_indices.normalized_difference(img, [2,7])

                if i < (t_len - 1): x_in[b, i, :, :, 0] = ndwi
                else:               x_ou[b,    :, :, 0] = ndwi

        return x_in, x_ou

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
        
        dataset = list(dataset.values())
        t_len = len(dataset[0])
        imgs = np.zeros((n_samples, t_len, img_shape[0], img_shape[1], img_shape[2]))

        for n in tqdm(range(n_samples), colour='blue'):
            imgs_path = dataset[n]
            for i, t in enumerate(imgs_path):
                img, _ = datareader.load(t)

                if normalize != None: img = normalizer.max_scaler(img, 10000)
                if img_shape != img.shape: img = cv2.resize(img, img_shape[:2])

                imgs[n, i, ...] = img[:,:,:img_shape[2]]


        return imgs

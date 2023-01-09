from tqdm.auto import tqdm
import numpy as np
import rasterio
import glob
import cv2
import os

from dataio.datareader import datareader

# Variable definitions
IMAGE_SHAPE = (300, 300, 13)
PATCH_SHAPE = ( 64,  64, 13)

dataset_name = 'DATASET-1' 
root         = os.path.join('.','datasets',dataset_name)
locations    = glob.glob(os.path.join(root, '*'))

# Loop through geo locations
for i, loc_path in enumerate(tqdm(locations)):
    images = glob.glob(os.path.join(loc_path, '*'))
    # Loop through images in a geo location
    for j, img_path in enumerate(images):
        # Load image
        img, meta = datareader.load(img_path)

        if img.shape != IMAGE_SHAPE: img = cv2.resize(img, IMAGE_SHAPE[:2], interpolation = cv2.INTER_NEAREST)
        counter = 0
        
        # Split patches
        for x in range(IMAGE_SHAPE[0]//PATCH_SHAPE[0]):
            for y in range(IMAGE_SHAPE[1]//PATCH_SHAPE[1]):
                # Extracting patch
                patch = img[x*PATCH_SHAPE[0]:(x+1)*PATCH_SHAPE[0], y*PATCH_SHAPE[1]:(y+1)*PATCH_SHAPE[1], :]
                # Create path
                patch_path = img_path.replace(loc_path.split(os.sep)[-1], 'p-{}-{}'.format(counter, loc_path.split(os.sep)[-1]))
                patch_path = patch_path.replace(dataset_name, dataset_name+'-v2')
                patch_path = patch_path.replace('gee_data_', '')

                #print(patch_path)

                #patch_path = os.path.join(patch_path, 'p-{}'.format(counter))
                os.makedirs(os.path.join(*patch_path.split(os.sep)[:-1]), exist_ok=True) 
                # Write image
                datareader.save(patch, patch_path, meta)
                counter += 1
            

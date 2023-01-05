import matplotlib.pyplot as plt
import numpy as np
import os

from dataio.datareader import datareader
from dataio.datahandler import datahandler

def plot_series(dataset_root_path, img_shape, bands, t_len = 30, n_samples = 5):
    '''
        This snippet will plot some samples from the dataset.

        Inputs:
        - dataset_root_path: system path to the dataset
        - t_len: lenght of the time series to be plotted
        - n_samples: number of samples to plot

        ------------------------------------------------------------------------
        Note the the plot will be save in ~/tmp/plt/dataset.png

    '''

    # Create folders for the plot
    save_path = os.path.join('tmp', 'plts')
    os.makedirs(save_path, exist_ok=True)

    # Load images from dataset
    d_handler = datahandler(dataset_root_path)
    imgs      = datareader.load_samples(d_handler.paths, n_samples, img_shape, normalize = True)

    # Create figure
    fig, axes = plt.subplots(nrows = n_samples, ncols = t_len, figsize = (3*t_len, 4*n_samples))

    for n in range(n_samples):
        for t in range(t_len):
            axes[n, t].imshow(imgs[n,t, :, :, bands], cmap='magma')

            axes[n, t].set(xticklabels=[])
            axes[n, t].set(yticklabels=[])
            axes[n, t].tick_params(bottom=False)
            axes[n, t].tick_params(left=False)
            
            axes[n, t].set_title('Time Step - {}'.format(t+1), fontsize = 20)

            if t==0:
                axes[n, t].set_ylabel('Sample - {}'.format(n+1), fontsize = 20)


    fig.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(save_path, 'dataset.png')) # Save figure
    plt.close()
    
    print('\t * Figure saved at "{}"'.format(save_path))

def plot_series2(dataset_root_path, img_shape, t_len = 30, n_samples = 50, n_display=5):
    '''
        This snippet will plot some samples from the dataset.

        Inputs:
        - dataset_root_path: system path to the dataset
        - t_len: lenght of the time series to be plotted
        - n_samples: number of samples to plot

        ------------------------------------------------------------------------
        Note the the plot will be save in ~/tmp/plt/dataset.png

    '''

    # Create folders for the plot
    save_path = os.path.join('tmp', 'plts')
    os.makedirs(save_path, exist_ok=True)

    # Load images from dataset
    d_handler = datahandler(dataset_root_path)
    imgs      = datareader.load_samples(d_handler.paths, n_samples, img_shape, normalize = True)

    # Create figure
    fig, axes = plt.subplots(nrows = n_display, ncols = t_len, figsize = (3*t_len, 4*n_display))

    #for n in range(n_samples):
    n = 0
    ct = 0
    while n < n_display:
        complete = True

        for t in range(t_len):
            rgb = np.zeros((img_shape[0], img_shape[1], 3))
            rgb[:,:,0] = imgs[ct,t, :, :, 3]
            rgb[:,:,1] = imgs[ct,t, :, :, 2]
            rgb[:,:,2] = imgs[ct,t, :, :, 1]

            rgb = 3.0*rgb
            rgb = np.clip(rgb, 0.0, 1.0)

            rgbnan = np.isnan(rgb[:,:,0])
            #print(rgbnan.any())
            if sum(sum(rgbnan)) > 10: complete = False

            axes[n, t].imshow(rgb)

            axes[n, t].set(xticklabels=[])
            axes[n, t].set(yticklabels=[])
            axes[n, t].tick_params(bottom=False)
            axes[n, t].tick_params(left=False)
        
        if complete == True: n=n+1
        ct += 1
            
            #axes[n, t].set_title('Time Step - {}'.format(t+1), fontsize = 20)

            #if t==0:
            #    axes[n, t].set_ylabel('Sample - {}'.format(n+1), fontsize = 20)


    fig.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(save_path, 'dataset2.png')) # Save figure
    plt.close()
    
    print('\t * Figure saved at "{}"'.format(save_path))

def plot_series3(dataset_root_path, img_shape, t_len = 30, n_samples = 50, n_display=5):
    '''
        This snippet will plot some samples from the dataset.

        Inputs:
        - dataset_root_path: system path to the dataset
        - t_len: lenght of the time series to be plotted
        - n_samples: number of samples to plot

        ------------------------------------------------------------------------
        Note the the plot will be save in ~/tmp/plt/dataset.png

    '''

    # Create folders for the plot
    save_path = os.path.join('tmp', 'plts')
    os.makedirs(save_path, exist_ok=True)

    # Load images from dataset
    d_handler = datahandler(dataset_root_path)
    imgs      = datareader.load_samples(d_handler.paths, n_samples, img_shape, normalize = True)

    # Create figure
    fig, axes = plt.subplots(nrows = n_display, ncols = t_len, figsize = (3*t_len, 4*n_display))

    #for n in range(n_samples):
    n = 0
    ct = 0
    while n < n_display:
        complete = True

        for t in range(t_len):
            rgb = np.zeros((img_shape[0], img_shape[1], 3))
            rgb[:,:,0] = imgs[ct,t, :, :, 3]
            rgb[:,:,1] = imgs[ct,t, :, :, 2]
            rgb[:,:,2] = imgs[ct,t, :, :, 1]

            rgb = 3.0*rgb
            rgb = np.clip(rgb, 0.0, 1.0)

            rgbnan = np.isnan(rgb[:,:,0])
            #print(rgbnan.any())
            if sum(sum(rgbnan)) > 10: complete = False

            axes[n, t].imshow(rgb)

            axes[n, t].set(xticklabels=[])
            axes[n, t].set(yticklabels=[])
            axes[n, t].tick_params(bottom=False)
            axes[n, t].tick_params(left=False)

            axes[n, t].set_title('Time Step - {}'.format(t+1), fontsize = 27)

            if t==0:
                axes[n, t].set_ylabel('Sample - {}'.format(n+1), fontsize = 27)
        
        if complete == True: n=n+1
        ct += 1
            
            


    fig.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(save_path, 'dataset3.png')) # Save figure
    plt.close()
    
    print('\t * Figure saved at "{}"'.format(save_path))
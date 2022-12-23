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
    plt.show()
    fig.savefig(os.path.join(save_path, 'dataset.png')) # Save figure
    plt.close()



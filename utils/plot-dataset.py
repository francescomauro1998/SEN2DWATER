import matplotlib.pyplot as plt
import numpy as np
import os

from dataio.datareader import datareader
from dataio.datahandler import datahandler

def plot_series(dataset_root_path, t_len = 30, n_samples = 5):
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
    save_path = os.path.join(tmp, plts)
    os.makedirs(save_path, exist_ok=True)

    # Load images from dataset
    d_handler = datahandler(dataset_root_path)
    imgs      = ldatareader.oad_samples(d_handler.paths, n_samples, (300,300,13), normalize = True)

    # Create figure
    fig, axes = plt.subplots(nrows = n_samples, ncols = t_len, figsize = (t_len, n_samples))

    for n in range(n_samples):
        for t in range(t_len):
            axes[n, t].imshow(imgs[n,t,...], cmap='magma')
            axes[n, t].axis(False)
            axes[n, t].set_title('Time Step {}'.format(t+1))

            if t==0:
                axes[n, t].set_ylabel('Sample {}'.format(n))


    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(save_path, 'dataset.png')) # Save figure
    plt.close()



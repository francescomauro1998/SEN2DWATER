import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os


from utils import plot_dataset


## Uncomment the following lines to plot some samples of the dataset
#plot_dataset.plot_series2('datasets/DATASET-1', (300,300,13), t_len = 39, n_samples = 50, n_display=10)
#plot_dataset.plot_series3('datasets/DATASET-1', (300,300,13), t_len = 39, n_samples = 50, n_display=10)

##############################################################################################################

## Ground Truths
gt_0           = plt.imread('tmp/LSTMv1/20230108-192300/res/gt/epoch-30/gt-0.png')[...,0]
gt_7           = plt.imread('tmp/LSTMv1/20230108-192300/res/gt/epoch-30/gt-7.png')[...,0]
gt_15          = plt.imread('tmp/LSTMv1/20230108-192300/res/gt/epoch-30/gt-15.png')[...,0]

## LSTM RESULTS
lstm_pr_0      = plt.imread('tmp/LSTMv1/20230108-192300/res/pr/epoch-30/pt-0.png')[...,0]
lstm_pr_7      = plt.imread('tmp/LSTMv1/20230108-192300/res/pr/epoch-30/pt-7.png')[...,0]
lstm_pr_15     = plt.imread('tmp/LSTMv1/20230108-192300/res/pr/epoch-30/pt-15.png')[...,0]

lstm_results   = pd.read_csv('tmp/LSTMv1/20230108-192300/res/df/epoch-30/results.csv', sep='\t')
bilstm_results = pd.read_csv('tmp/LSTMv1/20230108-192300/res/df/epoch-30/results.csv', sep='\t')
tdcnn_results  = pd.read_csv('tmp/LSTMv1/20230108-192300/res/df/epoch-30/results.csv', sep='\t')


fig, axes = plt.subplot_mosaic('ABCD;EFGH;IJKL;MMMM;NNNN;OOOO')#plt.subplots(nrows = 4, ncols = 4)

print(lstm_results.columns)
cmap = "viridis"

# Ground truth
axes['A'].imshow(gt_0,  cmap = mpl.colormaps[cmap])
axes['A'].axis(False)
axes['A'].set_title('Ground Truth')
axes['E'].imshow(gt_7,  cmap = mpl.colormaps[cmap])
axes['E'].axis(False)
axes['I'].imshow(gt_15, cmap = mpl.colormaps[cmap])
axes['I'].axis(False)

# ConvLSTM Prediction
axes['B'].imshow(lstm_pr_0,  cmap = mpl.colormaps[cmap])
axes['B'].axis(False)
axes['B'].set_title('ConvLSTM')
axes['F'].imshow(lstm_pr_7,  cmap = mpl.colormaps[cmap])
axes['F'].axis(False)
axes['J'].imshow(lstm_pr_15, cmap = mpl.colormaps[cmap])
axes['J'].axis(False)

# Bidirectional ConvLSTM Prediction
axes['C'].imshow(lstm_pr_0,  cmap = mpl.colormaps[cmap])
axes['C'].axis(False)
axes['C'].set_title('Bi-ConvLSTM')
axes['G'].imshow(lstm_pr_7,  cmap = mpl.colormaps[cmap])
axes['G'].axis(False)
axes['K'].imshow(lstm_pr_15, cmap = mpl.colormaps[cmap])
axes['K'].axis(False)

# Time Distributed CNN Prediction
axes['D'].imshow(lstm_pr_0,  cmap = mpl.colormaps[cmap])
axes['D'].axis(False)
axes['D'].set_title('TD-CNN')
axes['H'].imshow(lstm_pr_7,  cmap = mpl.colormaps[cmap])
axes['H'].axis(False)
axes['L'].imshow(lstm_pr_15, cmap = mpl.colormaps[cmap])
axes['L'].axis(False)

# MSE
axes['M'].plot(lstm_results['MSE'], '-*', label='ConvLSTM')
axes['M'].plot(bilstm_results['MSE'], '-*', label='Bi-ConvLSTM')
axes['M'].plot(tdcnn_results['MSE'], '-*', label='TD-CNN')
axes['M'].set_ylabel('MSE')
axes['M'].set_xticklabels([''])
axes['M'].legend(loc='upper right')

# SSIM
axes['N'].plot(lstm_results['SSIM'], '-*', label='ConvLSTM')
axes['N'].plot(bilstm_results['SSIM'], '-*', label='Bi-ConvLSTM')
axes['N'].plot(tdcnn_results['SSIM'], '-*', label='TD-CNN')
axes['N'].set_ylabel('SSIM')
axes['N'].set_xticklabels([''])
axes['N'].legend(loc='lower right')

# PSNR
axes['O'].plot(lstm_results['PNSR'], '-*', label='ConvLSTM')
axes['O'].plot(bilstm_results['PNSR'], '-*', label='Bi-ConvLSTM')
axes['O'].plot(tdcnn_results['PNSR'], '-*', label='TD-CNN')
axes['O'].set_ylabel('PSNR')
axes['O'].set_xlabel('Tests')
axes['O'].legend(loc='lower right')

#fig.tight_layout()
plt.show()

print('\n\n')
print('Averaged MSE  - ConvLSTM {} - Bi-ConvLSTM {} - TD-CNN {}'.format(lstm_results['MSE'].mean(),  bilstm_results['MSE'].mean(),  tdcnn_results['MSE'].mean()))
print('Averaged SSIM - ConvLSTM {} - Bi-ConvLSTM {} - TD-CNN {}'.format(lstm_results['SSIM'].mean(), bilstm_results['SSIM'].mean(), tdcnn_results['SSIM'].mean()))
print('Averaged PSNR - ConvLSTM {} - Bi-ConvLSTM {} - TD-CNN {}'.format(lstm_results['PNSR'].mean(), bilstm_results['PNSR'].mean(), tdcnn_results['PNSR'].mean()))

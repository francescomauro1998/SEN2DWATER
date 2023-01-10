import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os


##############################################################################################################

## Ground Truths
gt_0           = plt.imread('tmp/LSTMv1/20230108-192300/res/gt/epoch-30/gt-0.png' )[...,0]
gt_7           = plt.imread('tmp/LSTMv1/20230108-192300/res/gt/epoch-30/gt-7.png' )[...,0]
gt_15          = plt.imread('tmp/LSTMv1/20230108-192300/res/gt/epoch-30/gt-15.png')[...,0]

gt_0 = 2*gt_0 - 1
gt_7 = 2*gt_7 - 1
gt_15 = 2*gt_15 - 1

## LSTM RESULTS
lstm_pr_0      = plt.imread('tmp/LSTMv1/20230108-192300/res/pr/epoch-30/pt-0.png' )[...,0]
lstm_pr_7      = plt.imread('tmp/LSTMv1/20230108-192300/res/pr/epoch-30/pt-7.png' )[...,0]
lstm_pr_15     = plt.imread('tmp/LSTMv1/20230108-192300/res/pr/epoch-30/pt-15.png')[...,0]

lstm_pr_0   = 2*lstm_pr_0  - 1
lstm_pr_7   = 2*lstm_pr_7  - 1
lstm_pr_15  = 2*lstm_pr_15 - 1

## Bi-LSTM RESULTS ***
bilstm_pr_0      = plt.imread('tmp/BLSTMv1/20230109-044615/res/pr/epoch-35/pt-0.png' )[...,0]
bilstm_pr_7      = plt.imread('tmp/BLSTMv1/20230109-044615/res/pr/epoch-35/pt-7.png' )[...,0]
bilstm_pr_15     = plt.imread('tmp/BLSTMv1/20230109-044615/res/pr/epoch-35/pt-15.png')[...,0]

bilstm_pr_0   = 2*bilstm_pr_0  - 1
bilstm_pr_7   = 2*bilstm_pr_7  - 1
bilstm_pr_15  = 2*bilstm_pr_15 - 1

## TD-CNN RESULTS ***
tdcnn_pr_0      = plt.imread('tmp/ TDCNNv1/20230110-065019/res/pr/epoch-24/pt-0.png')[...,0]
tdcnn_pr_7      = plt.imread('tmp/ TDCNNv1/20230110-065019/res/pr/epoch-24/pt-7.png')[...,0]
tdcnn_pr_15     = plt.imread('tmp/ TDCNNv1/20230110-065019/res/pr/epoch-24/pt-15.png')[...,0]

tdcnn_pr_0   = 2*tdcnn_pr_0  - 1
tdcnn_pr_7   = 2*tdcnn_pr_7  - 1
tdcnn_pr_15  = 2*tdcnn_pr_15 - 1

## Numerical Results ***
lstm_results   = pd.read_csv('tmp/LSTMv1/20230108-192300/res/df/epoch-30/results.csv', sep='\t')
bilstm_results = pd.read_csv('tmp/BLSTMv1/20230109-044615/res/df/epoch-35/results.csv', sep='\t')
tdcnn_results  = pd.read_csv('tmp/ TDCNNv1/20230110-065019/res/df/epoch-24/results.csv', sep='\t')


##############################################################################################################
fig, axes = plt.subplot_mosaic('ABCD;EFGH;IJKL;MMMM;NNNN;OOOO')#plt.subplots(nrows = 4, ncols = 4)

cmap = "gray"

# Ground truth
axes['A'].imshow(gt_0,  cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['A'].axis(False)
axes['A'].set_title('Ground Truth')
axes['E'].imshow(gt_7,  cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['E'].axis(False)
axes['I'].imshow(gt_15, cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['I'].axis(False)

# ConvLSTM Prediction
axes['B'].imshow(lstm_pr_0,  cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['B'].axis(False)
axes['B'].set_title('ConvLSTM')
axes['F'].imshow(lstm_pr_7,  cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['F'].axis(False)
axes['J'].imshow(lstm_pr_15, cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['J'].axis(False)

# Bidirectional ConvLSTM Prediction
axes['C'].imshow(bilstm_pr_0,  cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['C'].axis(False)
axes['C'].set_title('Bi-ConvLSTM')
axes['G'].imshow(bilstm_pr_7,  cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['G'].axis(False)
axes['K'].imshow(bilstm_pr_15, cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['K'].axis(False)

# Time Distributed CNN Prediction
axes['D'].imshow(tdcnn_pr_0,  cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['D'].axis(False)
axes['D'].set_title('TD-CNN')
axes['H'].imshow(tdcnn_pr_7,  cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
axes['H'].axis(False)
axes['L'].imshow(tdcnn_pr_15, cmap = mpl.colormaps[cmap], vmin = -1, vmax = 1)
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
#axes['N'].legend(loc='upper left')

# PSNR
axes['O'].plot(lstm_results['PNSR'], '-*', label='ConvLSTM')
axes['O'].plot(bilstm_results['PNSR'], '-*', label='Bi-ConvLSTM')
axes['O'].plot(tdcnn_results['PNSR'], '-*', label='TD-CNN')
axes['O'].set_ylabel('PSNR')
axes['O'].set_xlabel('Tests')
#axes['O'].legend(loc='upper left')

#fig.tight_layout()
plt.show()

print('\n\n')
print('Mean of MSE  - ConvLSTM {} - Bi-ConvLSTM {} - TD-CNN {}'.format(lstm_results['MSE'].mean(),  bilstm_results['MSE'].mean(),  tdcnn_results['MSE'].mean()))
print('Mean of SSIM - ConvLSTM {} - Bi-ConvLSTM {} - TD-CNN {}'.format(lstm_results['SSIM'].mean(), bilstm_results['SSIM'].mean(), tdcnn_results['SSIM'].mean()))
print('Mean of PSNR - ConvLSTM {} - Bi-ConvLSTM {} - TD-CNN {}'.format(lstm_results['PNSR'].mean(), bilstm_results['PNSR'].mean(), tdcnn_results['PNSR'].mean()))

print('Std of MSE  - ConvLSTM {} - Bi-ConvLSTM {} - TD-CNN {}'.format(lstm_results['MSE'].std(),  bilstm_results['MSE'].std(),  tdcnn_results['MSE'].std()))
print('Std of SSIM - ConvLSTM {} - Bi-ConvLSTM {} - TD-CNN {}'.format(lstm_results['SSIM'].std(), bilstm_results['SSIM'].std(), tdcnn_results['SSIM'].std()))
print('Std of PSNR - ConvLSTM {} - Bi-ConvLSTM {} - TD-CNN {}'.format(lstm_results['PNSR'].std(), bilstm_results['PNSR'].std(), tdcnn_results['PNSR'].std()))
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os

class PlotterTensorboard(Callback):
    '''
        This class relates to the TensorBoard callbacks specified in train function above.
    '''

    def __init__(self, model, generator, log_path):
        self.generator  = generator
        self.model      = model
        self.log_path   = log_path
        self.writer     = tf.summary.create_file_writer(log_path)

    def on_epoch_end(self, epoch, logs={}):
        x_in, y_in = self.generator
        y_pr = self.model.predict(x_in, verbose=0)

        # Tensorboard visualization
        with self.writer.as_default():
            tf.summary.image(name='Ground Truth', data=(y_in*255).astype(np.uint8), step=epoch, max_outputs=3)
            tf.summary.image(name='Prediction',   data=(y_pr*255).astype(np.uint8), step=epoch, max_outputs=3)
        
        # Save results (images)
        gt_path = os.path.join(self.log_path, 'res', 'gt', 'epoch-{}'.format(epoch))
        pr_path = os.path.join(self.log_path, 'res', 'pr', 'epoch-{}'.format(epoch))

        os.makedirs(gt_path, exist_ok = True)
        os.makedirs(pr_path, exist_ok = True)

        psnr, ssim, mse,  = [], [], []

        for i in range(y_in.shape[0]):
            plt.imsave(os.path.join(gt_path,'gt-{}.png'.format(i)), (y_in[i,...,0]*255).astype(np.uint8))
            plt.imsave(os.path.join(pr_path,'pt-{}.png'.format(i)), (y_pr[i,...,0]*255).astype(np.uint8))
            # Save results (text)
            x = tf.cast(y_in[i,...], tf.float32)
            y = tf.cast(y_pr[i,...], tf.float32)
            m1 = tf.reduce_mean(tf.image.psnr(x, y, 2.0))
            m2 = tf.reduce_mean(tf.image.ssim(x, y, 2.0))
            m3 = tf.reduce_mean(tf.keras.metrics.mean_squared_error(x, y))
            
            psnr.append(m1.numpy())
            ssim.append(m2.numpy())
            mse.append(m3.numpy())

        
        # Save results (text)
        df_path = os.path.join(self.log_path, 'res', 'df', 'epoch-{}'.format(epoch))
        os.makedirs(df_path, exist_ok = True)

        tosave = {"PNSR":psnr, "SSIM":ssim, "MSE":mse}
        df = pd.DataFrame(tosave)
        df.to_csv(os.path.join(df_path, 'results.csv'), sep='\t')

        # Saving model
        model_path = os.path.join(self.log_path, 'model.h5')
        self.model.save(model_path)
            
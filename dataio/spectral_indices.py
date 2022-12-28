import numpy as np


class spectral_indices:

    @staticmethod
    def normalized_difference(img, bands):
        '''
            Calculates the normalized difference betweeen two bands.

            Inputs:
                - img: is a multispectral/iperspectral image. Must be a numpy
                       array (W,H,B) width W, height H and B bands
                - bands: a tuple of two elements corresponing to the indexes of the
                         bands to be used for the normalized_difference
            Output:
                - nd: the normalized difference of bands of img. See formula below. 
            ------------------------------------------------------------------------
            The formula applied is the following

                                nd = (b1 - b2)/((b1 + b2) + e)

            e ~= 0.00001 to stabilize division with weak denominator.
        '''
        
        # Select image bands
        b1 = img[:,:,bands[0]]
        b2 = img[:,:,bands[1]]

        # Check for NaN and remove them
#        b1[np.isnan(b1)] = 0
#        b2[np.isnan(b2)] = 0

        # Apply formula
        nd = (b1 - b2)/ ((b1 + b2) + 0.00001)

        return nd


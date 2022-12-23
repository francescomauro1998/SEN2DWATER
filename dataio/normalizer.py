import numpy as np

class normalizer:
    '''
        Class containing static methods for normalizing images
    '''

    @staticmethod
    def percentile_prescaler(data, perc=95, mmin = None):
        '''
            Clip data between minimum and maximum based on the percentile at perc
        
            
            Inputs:
                - data: a WxHxB image, with W width, H height and B bands            
                - perc: the percentile value
                - mmin (optional): the minimum
            Outputs:
                - data: normalized WxHxB image, with W width, H height and B bands     
        '''
        
        if mmin == None: mmin = np.min(data)
        
        mmax = np.percentile(data, perc)
        data = np.clip(data, mmin, mmax)
        
        return data
    
    @staticmethod
    def minmax_scaler(data, mmin=None, mmax=None, clip = [None, None], bandwise = False): 
        '''
            Apply the min max scaler to the input data. The formula is:
            
            out = (data - minimum)/(maximum - minimum + E)          (1)
            
            where E is a stabilizer term.
            
            
            Inputs:
                - data: a WxHxB image, with W width, H height and B bands
                - mmin (optional): the minimum in equation (1)
                - mmax (optional): the maximum in equation (1)
                - clip (optional): a list of two values used to constrain the image values 
                - bandwise: if true apply (1) to each band separately
            Outputs: 
                - data: normalized WxHxB image, with W width, H height and B bands      
        '''

        E = 0.001
        
        if bandwise:
            for b in range(data.shape[-1]):
                if mmin == None: mmin = np.min(data[:,:,b])
                if mmax == None: mmax = np.max(data[:,:,b])
                data[:,:,b] = (data[:,:,b] - mmin)/((mmax - mmin)+E)
        else:
            if mmin == None: mmin = np.min(data)
            if mmax == None: mmax = np.max(data)
            data = (data - mmin)/((mmax - mmin)+E)
        
        if clip != [None, None]: data = np.clip(data, clip[0], clip[1])
        
        return data

    @staticmethod
    def max_scaler(data, mmax=None, clip = [None, None], bandwise=False): 
        '''
            Apply the max scaler to the input data. The formula is:
            
            out = data/maximum          (1)
            
            
            Inputs:
                - data: a WxHxB image, with W width, H height and B bands            
                - mmax (optional): the maximum in equation (1)
                - clip (optional): a list of two values used to constrain the image values
                - bandwise: if true apply (1) to each band separately
            Outputs:
                - data: normalized WxHxB image, with W width, H height and B bands
        '''
        
        if bandwise:
            for b in range(data.shape[-1]):
                if mmax == None: mmax = np.max(data[:,:,b])
                data[:,:,b] = data[:,:,b]/mmax
        else:
            if mmax == None: mmax = np.max(data)
            data = data/mmax
            
        
        if clip != [None, None]: data = np.clip(data, clip[0], clip[-1])
        
        return data

    @staticmethod
    def std_scaler(data, mmean=None, sstd = None, clip = [None, None], bandwise=False):
        '''
            Apply the standardizer to the input data. The formula is:
            
            out = (data - mean)/standard deviation         (1)
             
            Inputs:
                - data: a WxHxB image, with W width, H height and B bands            
                - mmean (optional): the mean in equation (1)
                - sstd (optional): the standard deviation in equation (1)
                - clip (optional): a list of two values used to constrain the image values
                - bandwise: if true apply (1) to each band separately
            Outputs:
                - data: normalized WxHxB image, with W width, H height and B bands     
        '''
        if bandwise:
            for b in range(data.shape[-1]):
                if mmean == None: mmean = np.mean(data[:,:,b])
                if sstd == None: sstd = np.std(data[:,:,b])
                data[:,:,b] = (data[:,:,b] - mmean)/sstd
        else:
            if mmean == None: mmean = np.mean(data)
            if sstd == None: sstd = np.std(data)
            data = (data - mmean)/sstd
        
        if clip != [None, None]: data = np.clip(data, clip[0], clip[-1])
        
        return data

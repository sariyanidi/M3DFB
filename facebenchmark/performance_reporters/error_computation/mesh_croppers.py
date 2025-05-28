import numpy as np
import matplotlib.pyplot as plt

class BaseMeshCropper():
    
    def crop(X, Xlmks, Y, Ylmks):
        raise NotImplementedError("""This class is not implemented 
                                  (a virtual method has been called)""")


class PointBasedCropper(BaseMeshCropper):
    
    def __init__(self, opts):
        self.opts = opts
        
        # The options need to contain the indices of landmarks to be used for
        # reference (e.g., typically those that correspond to outer eye corners,
        # nose tip, mouth corner etc.)
        assert 'dist_threshold_ratio' in self.opts
        assert 'ref_lmk_index' in self.opts
        assert 'leyec_index' in self.opts
        assert 'reyec_index' in self.opts
    

    def crop(self, X, Xlmks, Y=None, Ylmks=None):
        
        dist_threshold = self.opts['dist_threshold']
        ref_ix = self.opts['ref_lmk_index']
        dists = np.sqrt(np.sum((X-Xlmks[ref_ix:ref_ix+1,:])**2, axis=1))
        iod = np.linalg.norm(Xlmks[self.opts['reyec_index'],:]-Xlmks[self.opts['leyec_index'],:])
        idx = np.where(dists<dist_threshold*iod)[0]
        return X[idx]
    

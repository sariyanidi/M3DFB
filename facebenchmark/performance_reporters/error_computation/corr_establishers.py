import numpy as np


class BaseCorrespondenceEstablisher():
    
    def __init__(self, opts):
        self.opts = opts
    
    def establish_correspondence(self, X, Y, lmk_indices=None):
        raise NotImplementedError("""This class is not implemented 
                                      (a virtual method has been called)""")



class ChamferCorrespondence(BaseCorrespondenceEstablisher):
    
    def establish(self, X, Y):
        """
        Parameters
        ----------
        X : ndarray
            `N x 3` array containing the reference mesh.
        Y : ndarray
            `N x 3` array containing the mesh for which correspondences will be found.
        
        
        Returns
        -------
        pidx : ndarray
            `N x 1` array of integers, containing indices of the corresponding points,
            such that `X[i]` correspondgs to `Y[pidx[i]]`.

        """
        N = X.shape[0]
        pidx = np.zeros(N)
        
        for i in range(N):
            pidx[i] = np.argmin(np.sqrt((((X[i,:]-Y)**2).sum(axis=1))))
        
        return pidx.astype(int)

    
class IdentityCorrespondence(BaseCorrespondenceEstablisher):
    
    def establish(self, X, Y):
        """
        Parameters
        ----------
        X : ndarray
            `N x 3` array containing the reference mesh.
        Y : ndarray
            `N x 3` array containing the mesh for which correspondences will be found.
        
        
        Returns
        -------
        pidx : ndarray
            `N x 1` array of integers, containing indices of the corresponding points,
            such that `X[i]` correspondgs to `Y[pidx[i]]`.

        """
        pidx = np.arange(X.shape[0])
        
        return pidx.astype(int)

    


    


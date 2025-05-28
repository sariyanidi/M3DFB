import copy
import numpy as np
from scipy import interpolate
from scipy.sparse.linalg import spsolve 
from scipy.sparse import linalg as splinalg
from scipy import sparse

from facebenchmark import compute_landmark_base_vertex_weights

class BaseCorrector():
    
    def __init__(self, opts):
        self.opts = opts
    
    def align(self, R, G):
        raise NotImplementedError("""This class is not implemented 
                                  (a virtual method has been called)""")



class TopologyConsistencyCorrector(BaseCorrector):
    
    def __init__(self, opts, mm):
        super().__init__(opts)
        
        default_opts = {
            "correction_strategy": "pair",
            "weight_power": "sqrt", 
            "weight_strategy": "mixed"
        }
        
        self.opts = {**default_opts, **opts}
        self.weights = compute_landmark_base_vertex_weights(mm, self.opts['weight_power'], self.opts['weight_strategy'])
        self.lmk_indices = mm['lmk_indices']

    
    # def apply_correction(self, R2_, G_, Glmks, pidx):
    # def correct(self, R2_, G_, Glmks, pidx):
    def correct(self, X, Y):
        
        N = Y.shape[0]
        updates = []
        
        for dim in range(3):
            if self.opts['correction_strategy'] == 'trace':
                r = X[:,dim].reshape(-1,1)
                g = Y[:,dim].reshape(-1,1)
                e = np.ones(r.shape)
                
                gr = g-r
                r = e*np.sum(gr)-N*gr
                rT = r.T
                
                a = 2*(N-1)
                b = -2
                
                ads = self.weights*N*2+a 
                
                (diags, off_diags) = build_cholseky_factor(ads, b, N)
                c = -2*rT
                
                y = solve_lower_tri(diags, off_diags, -c.T)
                dx_star = solve_upper_tri(diags, off_diags, y)
                updates.append(dx_star)
    
            elif self.opts['correction_strategy'] == 'pair':
                
                r0 = X[:,dim] #.reshape(-1,1)
                g0 = Y[:,dim]
    
                rix = np.argsort(r0)
                r = copy.deepcopy(r0)[rix]
                g = copy.deepcopy(g0)[rix]
                
                c = np.zeros(N)
                c[0] = -(g[0]-g[1]-(r[0]-r[1]))
                c[-1] = -(g[-1]-g[-2]-(r[-1]-r[-2]))
                
                for i in range(1,N-1):
                    c[i] = -(2*g[i]-g[i-1]-g[i+1] - 2*r[i]+r[i-1]+r[i+1])
                
                d1 = -np.ones(N)#+weights[:N]
                d0 = 2 * np.ones(N)+2*(self.weights[rix])#/2
                A = (1./2)*sparse.spdiags([d1,d0,d1], [-1,0,1],N,N,format='csc')
                L = sparse_cholesky(A).T
                b = spsolve(L.T, c)
                y = spsolve(L, b)
                irix = np.empty_like(rix)
                irix[rix] = np.arange(rix.size)
                x = y[irix]
                updates.append(x)
        
        dG = np.array(updates).T
        
        return dG
    
    
def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
    n = A.shape[0]
    LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition
    # if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
    return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5) )



def build_cholseky_factor(ads, b, N):
    diags = []
    off_diags = []
    
    sum2_off_diags = 0
    for i in range(N):
        diags.append(np.sqrt(ads[i]-sum2_off_diags))
        if i == N-1:
            break
        off_diags.append((b-sum2_off_diags)/diags[-1])
        sum2_off_diags += off_diags[-1]*off_diags[-1]
    return (diags, off_diags)


def solve_lower_tri(alphas, betas, b):
    N = len(alphas)
    y = np.zeros(N)
    
    prevsum = 0
    for i in range(0,N):
        y[i] = (b[i]-prevsum)/alphas[i]
        if i != N-1:
            prevsum += betas[i]*y[i]
            
    return y


def solve_upper_tri(alphas, betas, y):
    N = len(alphas)
    x = np.zeros(N)
    
    xsum = 0
    betas.append(0)
    for i in reversed(range(0,N)):
        x[i] = (y[i]-xsum*betas[i])/alphas[i]
        xsum += x[i]
            
    return x



import copy
import numpy as np
import cvxpy as cp
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import spsolve 
from scipy.spatial import Delaunay


class BaseNonrigidAligner():
    
    def __init__(self, opts):
        self.opts = opts
    
    def align(self, R, G, Glmks, lmk_indices):
        raise NotImplementedError("""This class is not implemented 
                                  (a virtual method has been called)""")


class LandmarkBasedElasticAligner(BaseNonrigidAligner):
    
    def __init__(self, opts):
        super().__init__(opts)
        default_opts = { 'sel_lmk_ids': '0-51'}
        self.opts = {**default_opts, **opts}

        if self.opts['sel_lmk_ids'] == '0-51':
            self.sel_lmk_ids = np.arange(0,51).tolist()
        
        
    def align(self, R, G, Glmks, lmk_indices):

        lmk_indices = np.array(lmk_indices)
        
        Dx = squareform(pdist(R)).T
        Dx = Dx[:,lmk_indices]
        
        for j in range(Dx.shape[1]):
            Dx[:,j] = 1-Dx[:,j]/Dx[:,j].max()
        
        Dx = Dx**self.opts['gamma']
        Dy = Dx
        Dz = Dx
        
        Dxl = copy.deepcopy(Dx)[lmk_indices[self.sel_lmk_ids],:]
        Dyl = copy.deepcopy(Dy)[lmk_indices[self.sel_lmk_ids],:]
        Dzl = copy.deepcopy(Dz)[lmk_indices[self.sel_lmk_ids],:]
        
        bxl = Glmks[self.sel_lmk_ids,0]-R[lmk_indices[self.sel_lmk_ids],0]
        byl = Glmks[self.sel_lmk_ids,1]-R[lmk_indices[self.sel_lmk_ids],1]
        bzl = Glmks[self.sel_lmk_ids,2]-R[lmk_indices[self.sel_lmk_ids],2]
                
        dxu = self.solve_optimization_problem(Dx, Dxl, bxl)
        dyu = self.solve_optimization_problem(Dy, Dyl, byl)
        dzu = self.solve_optimization_problem(Dz, Dzl, bzl)
        
        N = R.shape[0]
        R2 = np.zeros((N,3))
        
        R2[:,0] = R[:,0] + Dx@dxu
        R2[:,1] = R[:,1] + Dy@dyu
        R2[:,2] = R[:,2] + Dz@dzu
        
        return R2
    
    
    def solve_optimization_problem(self, D, Dl, b):
        np.random.seed(1907)
        m = Dl.shape[0]
        n = Dl.shape[1]
        bmax = np.max(np.abs(b))
        #e2max = 0.2*emax;       
        
        Dsub = D[::10,:]
                
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(Dl@x-b))
        # constraints = [Dl@x-b <= e, E@x-f <= e2]
        # constraints = [ Dl@x-b <= emax, emin <= Dl@x-b]
        constraints = []
        
        
        constraints.append(cp.norm_inf(Dsub@x)  <= bmax)
        prob = cp.Problem(objective, constraints)
        
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve(solver='ECOS')
        # The optimal value for x is stored in `x.value`.
        # print(x.value)
        # The optimal Lagrange multiplier for a constraint is stored in
        # `constraint.dual_value`.
        # print(constraints[0].dual_value)
        
        return x.value
    



  
  # if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
    # return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5) )
  # else:
    # sys.exit('The matrix is not positive definite')
  
    
  

"""
Implementation based on code from: https://github.com/czh-98/REALY/blob/master/utils/NICP.py
However, we replaced the solution of the sparse system (see spsolve()) with our
alternative, because the original code uses the scikit-sparse package, which is
difficult to install without conda
"""
class NonrigidICPAligner(BaseNonrigidAligner):
    
    def __init__(self, opts):
        super().__init__(opts)
        
        default_opts = { 'epsilon':1.0, 'gamma':1, 'alpha':50, 'prealign_ELR': 0}
        self.opts = {**default_opts, **opts}
        
        self.epsilon = self.opts['epsilon']  # optimization ending threshold
        self.gamma = float(self.opts['gamma'])  # stiffness loss weight for translation
        self.alpha = float(self.opts['alpha'])  # exponential decay for each step
        
        if self.opts['prealign_ELR']:
            self.prealigner = LandmarkBasedElasticAligner({'gamma': 1})
    
    @staticmethod
    def find_nearest_neighbors(ver_src, ver_dst, n=100):
        # kd-tree to find NN
        kd_tree = scipy.spatial.KDTree(ver_dst.T, n)
        nn_distances, nn_indices = kd_tree.query(ver_src.T, 1, p=2)
        nn_indices = np.array(nn_indices, np.int32)
        nn_distances = np.array(nn_distances, np.float32)
        return nn_indices, nn_distances  

    
    @staticmethod
    def triangles_to_edge_vertex_adjacent_matrix(triangles):  
        p_list = []
        tri_t = triangles.T
        for v1, v2, v3 in tri_t:
            if v1 < v2:
                p_list.append([v1, v2])
            else:
                p_list.append([v2, v1])

            if v1 < v3:
                p_list.append([v1, v3])
            else:
                p_list.append([v3, v1])

            if v2 < v3:
                p_list.append([v2, v3])
            else:
                p_list.append([v3, v2])

        p = np.stack(p_list, axis=1)

        pair_list = np.split(p, p.shape[1], axis=1)
        pair_tuple_list = [(int(elem[0]), int(elem[1])) for elem in pair_list]
        pair_set = set(pair_tuple_list)
        pair_array = np.array(list(pair_set))
        edge_numbers = np.arange(len(pair_array))

        index_i = np.concatenate([edge_numbers] * 2, axis=0)
        index_j = np.concatenate([pair_array[:, 0], pair_array[:, 1]], axis=0)
        data = np.ones([len(pair_set)])
        data = np.concatenate([data, -data], axis=0)

        ev_adj = scipy.sparse.csc_matrix((data, (index_i, index_j)), shape=[len(edge_numbers), np.amax(triangles) + 1])
        return ev_adj

    @staticmethod
    def sparse_matrix_from_vertices(cur_src):
        # cur_src: 3xN
        one_row = np.ones([1, cur_src.shape[1]], np.float32)
        cur_src_ = np.concatenate([cur_src, one_row], axis=0)

        Di = np.arange(cur_src.shape[1])
        Dj = np.stack([Di * 4, Di * 4 + 1, Di * 4 + 2, Di * 4 + 3], axis=0)
        Dj = np.reshape(Dj, [-1])
        Di = np.stack([Di] * 4, axis=0)
        Di = np.reshape(Di, [-1])
        cur_src_vector = np.reshape(cur_src_, [-1])
        D = scipy.sparse.csc_matrix(
            (cur_src_vector, (Di, Dj)), shape=[cur_src.shape[1], cur_src.shape[1] * 4], dtype=np.float32
        )
        return D
    
    @staticmethod
    def spsolve(sparse_A, dense_b):
        from scipy.sparse import csc_matrix
        
        # # This was the old way of doing, we don't do this anymore because the
        # # package scikit-sparse causes dependency issues and is difficult to install
        # # with pip
        # from sksparse.cholmod import cholesky_AAt
        # factor = cholesky_AAt(sparse_A.T)
        # x = factor(sparse_A.T.dot(dense_b)).toarray()
        
        LU = scipy.sparse.linalg.splu(sparse_A.T@sparse_A, diag_pivot_thresh=0)
        
        P1 = csc_matrix((np.ones(LU.L.shape[0]), (LU.perm_r, np.arange(LU.L.shape[0])))).T
        P2 = csc_matrix((np.ones(LU.L.shape[0]), (np.arange(LU.L.shape[0]), LU.perm_c))).T
        
        btilde = P1.T @ (sparse_A.T@dense_b)

        z = spsolve(LU.L, btilde)
        y = spsolve(LU.U, z)
        x = (P2.T@y).toarray()
        
        return x

    def align(self, source_points, target_points, source_point_lmks=None, lmk_indices=None):
        
        if self.opts['prealign_ELR']:
            source_points = self.prealigner.align(source_points, target_points, source_point_lmks, lmk_indices)

        source_tri = Delaunay(source_points[:, :2]).simplices.T
        source = source_points.T
        target = target_points.T
        
        # G_: weighting for balancing rotation and translation in stiffness loss
        G = np.diag(np.array([1.0, 1.0, 1.0, self.gamma], np.float32))

        # homogeneous ver_src
        # M: build a node-arc incidence matrix for M
        M = self.triangles_to_edge_vertex_adjacent_matrix(source_tri)  # E x N

        # A1: large matrix for complete cost function
        A1 = scipy.sparse.kron(M, G)  # 4E x 4N

        # B1: large matrix for complete cost function
        B1 = np.zeros([A1.shape[0], 3], np.float32)
        B1 = scipy.sparse.csc_matrix(B1)

        cur_src = source
        cur_X = np.zeros([cur_src.shape[1] * 4, 3])
        X = np.ones([cur_src.shape[1] * 4, 3])  # initialization

        for decay in range(3):
            # print(decay)
            cur_alpha = self.alpha * np.exp(-0.5 * decay)
            epsilon = self.epsilon * pow(0.5, decay)
            step = 0
            while True:
                delta = np.linalg.norm(X - cur_X)
                # print(delta)
                if delta <= epsilon:
                    break

                cur_X = X.copy()

                # find nn
                nn_indices, nn_distances = self.find_nearest_neighbors(cur_src, target)
                cur_nn_ver_dst = target[:, nn_indices]

                cur_D = self.sparse_matrix_from_vertices(cur_src)  # N x 4N
                threshold = max(np.mean(nn_distances) * 2, 1)

                cur_weight = (nn_distances < threshold).astype(np.float32)

                A2 = cur_D.multiply(cur_weight[:, np.newaxis])
                B2 = cur_nn_ver_dst.T
                B2 = np.multiply(B2, cur_weight[:, np.newaxis])

                # convert to sparse matrix
                A2 = scipy.sparse.csc_matrix(A2)
                B2 = scipy.sparse.csc_matrix(B2)

                A = scipy.sparse.vstack([A1.multiply(cur_alpha), A2])
                B = scipy.sparse.vstack([B1, B2])

                X = self.spsolve(A, B)

                cur_src = cur_D * X
                cur_src = cur_src.transpose()
                step += 1

        return cur_src.T
    
    
    

if __name__ == "__main__":
    from facebenchmark.performance_reporters.error_computation.rigid_aligners import ICPRigidAligner, LandmarkBasedRigidAligner
    # icp = ICPRigidAlignment({})
    rigid = LandmarkBasedRigidAligner({})
    # nicp = NICP_with_keypoints({})
    nicp = NonrigidICPAligner({'alpha': 1, 'epsilon': 1})

    li = [19106,19413,19656,19814,19981,20671,20837,20995,21256,21516,8161,8175,8184,8190,6758,7602,8201,8802,9641,1831,3759,5049,6086,4545,3515,10455,11482,12643,14583,12915,11881,5522,6154,7375,8215,9295,10523,10923,9917,9075,8235,7395,6548,5908,7264,8224,9184,10665,8948,8228,7508]

    G = np.loadtxt('./data/BFMsynth/Gmeshes/id0004.txt')
    Glmks = np.loadtxt('./data/BFMsynth/Gmeshes/id0004.lmks')
    R = np.loadtxt('./data/BFMsynth/Rmeshes/BFM/p23470/Deep3DFace_050/id0004.txt')

    li68 = [748,595,1250,1350,1352,1356,1359,1382,1537,1541,1549,1552,1556,1558,602,23,78,597,1339,4,1594,720,50,1732,0,705,25,1474,1478,1439,1480,1075,1112,1476,450,407,884,863,926,1538,785,825,1705,256,239,226,117,115,1047,1131,1097,1467,435,469,379,548,587,1441,1245,1206,1163,1103,1459,441,378,582,1447,1240]
    li = np.array(li68)[17:].tolist()


    G = np.loadtxt('/online_data/3Dfacebenchmark/FLAME_synth/Gmeshes/id0003.txt')
    Glmks = np.loadtxt('/online_data/3Dfacebenchmark/FLAME_synth/Gmeshes/id0003.lmks')
    # R = np.loadtxt('/online_data/3Dfacebenchmark/FLAME_synth/Rmeshes/FLAME/face/EMOCA_001/id0003.txt')
    R = np.loadtxt('/online_data/3Dfacebenchmark/FLAME_synth/Rmeshes/FLAME/face/MICA_001/id0003.txt')

    import matplotlib.pyplot as plt

    R, Rlmks = rigid.align(R, G, R[li], Glmks)
    R[:,0] *= 0.89
    Rlmks[:,0] *= 0.89

    plt.plot(G[:,0], G[:,1], '.')
    plt.plot(R[:,0], R[:,1], '.')
    R2 = nicp.align(R, G[:1787],Glmks, li)

    
    


import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import Delaunay

class BaseRigidAligner():
    
    def align(X, Y, Xlmks, Ylmks):
        raise NotImplementedError("""This class is not implemented 
                                  (a virtual method has been called)""")


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Taken from https://github.com/patrikhuber/fg2018-competition
    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX
        # print(Y)

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform



class LandmarkBasedRigidAligner(BaseRigidAligner):
    
    def __init__(self, opts):
        
        default_opts = {"ref_lmk_indices": [13, 19, 28, 31, 37]}
        self.opts = {**default_opts, **opts}
        
        # The options need to contain the indices of landmarks to be used for
        # reference (e.g., typically those that correspond to outer eye corners,
        # nose tip, mouth corner etc.)
        assert 'ref_lmk_indices' in self.opts
    

    def align(self, X, Y, Xlmks, Ylmks):
        
        ref_ix = self.opts['ref_lmk_indices']
        
        _, __, tform = procrustes(Ylmks[ref_ix,:], Xlmks[ref_ix,:])

        X = tform['scale']*(X @ tform['rotation'])+tform['translation']
        Xlmks = tform['scale']*(Xlmks @ tform['rotation'])+tform['translation']
        
        return X, Xlmks



class ICPRigidAligner(BaseRigidAligner):
    
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.land_align = LandmarkBasedRigidAligner({"ref_lmk_indices": [13, 19, 28, 31, 37]})
        
    def align(self, source_points, target_points, source_lmks_points, target_lmks_points):
        
        source_points, source_lmks_points = self.land_align.align(source_points, target_points, 
                                                                  source_lmks_points, target_lmks_points)
        
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(np.asarray(target_points))

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(np.asarray(source_points))
        
        source_lmks = o3d.geometry.PointCloud()
        source_lmks.points = o3d.utility.Vector3dVector(np.asarray(source_lmks_points))

        # Choose this number because from experimentation it gave the best results
        threshold = 1000  # Adjust the threshold as needed
        
        # print("Apply point-to-point ICP")
        icp_values = o3d.pipelines.registration.registration_icp(
            source, target, threshold)

        # print("Transformation matrix:")
        # print(icp_values.transformation)

        source_new = source.transform(icp_values.transformation)
        source_new_lmks = source_lmks.transform(icp_values.transformation)
        
        return np.asarray(source_new.points), np.asarray(source_new_lmks.points)




import os
import itertools
import numpy as np

import matplotlib as mpl
import matplotlib.cm as cm


from facebenchmark import BaseReporter
import matplotlib.pyplot as plt


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'cubic',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image




class HeatmapVisualizer(BaseReporter):
    
    def compute_all_heatmaps(self, err_computers, rec_method, Nsubj=100):
        for subj_id in range(Nsubj):
            if not self.subject_exists(subj_id):
                continue
            self.compute_heatmaps(subj_id, err_computers, rec_method, save_figure=True)


    
    
    def compute_heatmap(self, subj_id, rec_method_full, err_computer, norm_coef, millimeter=True):
        
        mm_name = rec_method_full.split('/')[0]
        mm = self.mms_info[mm_name]

        err_ptwise = self.compute_pervertex_error(subj_id, rec_method_full, err_computer)
        
        R, Rlmks = self.get_Rdata(subj_id, rec_method_full)
        
        # @@@ This needs to be replaced with each methods landmark ID
        iod = np.linalg.norm(Rlmks[mm['leye_oc_rel_index'],:]-Rlmks[mm['reye_oc_rel_index'],:])
        R /= iod

        # R[:,1] *= -1
        
        coef = 50
        x = ((R[:,0]-R[:,0].min())*coef).round().reshape(-1,).astype(int)
        y = ((R[:,1]-R[:,1].min())*coef).round().reshape(-1,).astype(int)
        
        H = np.nan*np.ones((y.max(), x.max()))
        
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        m = cm.ScalarMappable(norm=norm,cmap='jet')

        for i in range(len(x)):
            if self.cpts is None:
                H[H.shape[0]-(y[i]+1), x[i]-1] = err_ptwise[i]
            else:
                if i in self.cpts:
                    H[H.shape[0]-(y[i]+1), x[i]-1] = err_ptwise[i]
                else:
                    H[H.shape[0]-(y[i]+1), x[i]-1] = 0
                    
        D = interpolate_missing_pixels(H,np.isnan(H))
        D = D/norm_coef
                        

        if self.cpts is not None:
            Hmask = np.nan*np.ones((y.max(), x.max()))
    
            for i in range(len(x)):
                if self.cpts is None:
                    Hmask[Hmask.shape[0]-(y[i]+1), x[i]-1] = 1
                else:
                    if i in self.cpts:
                        Hmask[Hmask.shape[0]-(y[i]+1), x[i]-1] = 1
                    else:
                        Hmask[Hmask.shape[0]-(y[i]+1), x[i]-1] = 0

            Dmask = interpolate_missing_pixels(Hmask,np.isnan(Hmask))
        
        Dc = m.to_rgba(D)[:,:,:3]
        for i, j in itertools.product(range(Dc.shape[0]), range(Dc.shape[1])):
            if self.cpts is not None:
                if D[i,j] == 0 or Dmask[i,j] < 0.95:
                    Dc[i,j,0] = 1
                    Dc[i,j,1] = 1
                    Dc[i,j,2] = 1
            else:
                if D[i,j] == 0:
                    Dc[i,j,0] = 1
                    Dc[i,j,1] = 1
                    Dc[i,j,2] = 1

        return Dc
    

    def compute_heatmaps(self, subj_id, rec_method_full, save_figure=False):
        assert subj_id in self.subj_ids
        """
        subj_fname = 'subj%03d' % subj_id
        figdir = self.figdir_root + "/heatmaps/" + err_computers[0].get_key(self.landmark_computer) + "_" + self.aligner + self.tail +"/" 
        figpath_png = '%s/%s-%s.png' % (figdir, rec_method, subj_fname)   
        """
        #if os.path.exists(figpath_png):
        #    print(figpath_png)
        #    return True
        
        mm = rec_method_full.split('/')[0]
        errs_ptwise = list(self.compute_pervertex_error_all_computers(subj_id, rec_method_full).values())
        err_max = max([np.percentile(err_ptwise, 95) for err_ptwise in errs_ptwise])
        

        Ds = [self.compute_heatmap(subj_id, rec_method_full, ec, err_max) for ec in self.error_computers[mm]]
        
        plt.clf()
        for i in range(len(Ds)):
            plt.subplot(1, len(Ds), i+1)
            plt.imshow(Ds[i])
            plt.axis('off')
            
        """
        if save_figure:
            assert self.figdir_root is not None
            
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            
            plt.savefig(figpath_png, dpi=100, bbox_inches='tight', pad_inches=0)
            """


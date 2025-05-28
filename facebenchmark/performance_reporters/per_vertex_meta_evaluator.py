import numpy as np

import matplotlib.pyplot as plt
from facebenchmark import MetaEvaluator
from sklearn.linear_model import LinearRegression


class PerVertexMetaEvaluator(MetaEvaluator):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        default_opts = {"Npoints": 50000,
                        "alpha": 0.002,
                        "save_filepath": None}
        
        self.opts = {**default_opts, **self.opts}
        
        
    def produce(self):
        
        self.compute_all_pervertex_errors()
                
        mm0 = self.rec_methods[0].split('/')[0]
        
        true_err_computer = self.error_computers[mm0][0]
        est_err_computer = self.error_computers[mm0][1]
        
        true_errs = np.array([])
        est_errs = np.array([])
        
        for rec_method_full in self.rec_methods:
            for subj_id in self.subj_ids:
                true_errs = np.concatenate((true_errs,self.compute_pervertex_error(subj_id, rec_method_full, true_err_computer)))
                est_errs = np.concatenate((est_errs,self.compute_pervertex_error(subj_id, rec_method_full, est_err_computer)))
                
        idx = np.arange(true_errs.shape[0]).astype(int)
        np.random.shuffle(idx)
        
        true_errs = true_errs[idx]
        est_errs =  est_errs[idx]
        
        model = LinearRegression(fit_intercept=False)
        model.fit(np.array(true_errs).reshape(-1,1), np.array(est_errs).reshape(-1,1))
        r_squared = model.score(np.array(true_errs).reshape(-1,1), np.array(est_errs).reshape(-1,1))
        slope = model.coef_[0]
        
        mn = 0.04
        mx = 0.18
        d = mx-mn
        
        
        plt.text(mn+d*0.07, mx-d*0.1, r"slope=%.2f" % slope,
                color="black", fontsize=12,
                horizontalalignment="left", verticalalignment="center",
                bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))
        
        plt.text(mn+d*0.07, mx-d*0.20, r"$R^2$=%.2f" % r_squared,
                color="black", fontsize=12,
                horizontalalignment="left", verticalalignment="center",
                bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))
        plt.plot(true_errs[:self.opts['Npoints']], est_errs[:self.opts['Npoints']], 'o', alpha=self.opts['alpha'])
        
        if self.opts['save_filepath'] is not None:
            plt.savefig(self.opts['save_filepath'])

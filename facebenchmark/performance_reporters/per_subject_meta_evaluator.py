import numpy as np

import matplotlib.pyplot as plt
from facebenchmark import MetaEvaluator
from sklearn.linear_model import LinearRegression


class PerSubjectMetaEvaluator(MetaEvaluator):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        default_opts = {"Npoints": 50000,
                        "alpha": 0.01,
                        "persubj_statistic": "mean"}
        
        self.opts = {**default_opts, **self.opts}
        
        # The plots here are based on comparing two error computers (err1 vs err2)
        assert len(self.error_computer_names) == 2

        
        
    def produce(self):
        
        self.compute_all_pervertex_errors()
        
        colors = list(iter([plt.cm.tab10(i) for i in range(10)]))

        legends = []
        all_true_errs = []
        all_est_errs = []
        for ri, rec_method_full in enumerate(self.rec_methods):
            mm = rec_method_full.split('/')[0]
            
            true_err_computer = self.error_computers[mm][0]
            est_err_computer = self.error_computers[mm][1]
            
            true_errs = []
            est_errs = []
            for subj_id in self.subj_ids:
                true_err = self.compute_pervertex_error(subj_id, rec_method_full, true_err_computer)
                est_err = self.compute_pervertex_error(subj_id, rec_method_full, est_err_computer)
                if self.opts['persubj_statistic'] == 'mean':
                    true_errs.append(np.mean(true_err))
                    est_errs.append(np.mean(est_err))
                elif self.opts['persubj_statistic'] == 'median':
                    true_errs.append(np.median(true_err))
                    est_errs.append(np.median(est_err))
            
            method_name = rec_method_full.split('/')[1]
            legends.append(method_name)
            
            mdl = LinearRegression(fit_intercept=False)
            mdl.fit(np.array(true_errs).reshape(-1,1), np.array(est_errs))
            slope = mdl.coef_[0]
        
            plt.scatter(true_errs, est_errs, alpha=0.35, color=colors[ri], s=24)
            

            print('%s: %.2f' % (method_name, slope))
            plt.plot([-1, 1], [-1*slope, slope*1], color=colors[ri], alpha=1)
            all_true_errs += true_errs
            all_est_errs += est_errs
            # slopes.append(slope)
        
        minx, maxx = min(all_true_errs), max(all_true_errs)
        miny, maxy = min(all_est_errs), max(all_est_errs)
        
        plt.xlim((minx, maxx))
        plt.ylim((miny, maxy))
        
import itertools
import numpy as np

from facebenchmark import BaseReporter
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats

class CorrelationReporter(BaseReporter):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        default_opts = {"overall_statistic": "mean", 
                        "persubj_statistic": "mean", 
                        "layout": "rec_methods/err_computers", 
                        "show_row_headers": True,
                        "show_col_headers": True,
                        "round_digits": 2,
                        "divide_error_by": 1.0, 
                        "delimiter": "\t"}
        
        self.opts = {**default_opts, **self.opts}
    
    
    def produce(self):
        
        self.compute_all_pervertex_errors()
        
        mm0 = self.rec_methods[0].split('/')[0]
        
        ref_err_computer = self.error_computers[mm0][0]
        
        alpha = 0.9
        Nplots = len(self.error_computers[mm0][1:])
        plt.figure(figsize=(alpha*2.01*Nplots, alpha*2.4))
        for ei, ec in enumerate(self.error_computers[mm0][1:]):
            plt.subplot(1, Nplots, ei+1)
            refs = []
            ests = []
            for ri, rec_method_full in enumerate(self.rec_methods):
                
                ref_errs = []
                est_errs = []
                
                for subj_id in self.subj_ids:
                    ref_errs.append(self.compute_pervertex_error(subj_id, rec_method_full, ref_err_computer).mean())
                    est_errs.append(self.compute_pervertex_error(subj_id, rec_method_full, ec).mean())
                
                refs.append(np.mean(ref_errs)/self.opts['divide_error_by'])
                ests.append(np.mean(est_errs)/self.opts['divide_error_by'])

            plt.scatter(refs, ests, alpha=0.4)
            plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
                
            mnx = np.min(refs)
            mny = np.min(ests)
            
            dx = np.max(refs)-mnx
            dy = np.max(ests)-mny
            
            corr = scipy.stats.pearsonr(refs, ests)[0]
            corr_top5 = scipy.stats.pearsonr(refs[:5], ests[:5])[0]
            
            fs = 11

            plt.text(mnx+dx*0.01, mny+dy*0.95, r"$\rho_{Top5}$=%.2f" % corr_top5,
                    color="black", fontsize=fs,
                    horizontalalignment="left", verticalalignment="center",
                    bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))
            
            
            plt.text(mnx+dx*0.01, mny+dy*0.80, r"$\rho$=%.2f" % corr,
                    color="black", fontsize=fs,
                    horizontalalignment="left", verticalalignment="center",
                    bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))
            
            
                
            plt.title(f'{ec.name}')
            plt.xlabel('True err.')
            if ei == 0:
                plt.ylabel('Estimated err.')
        
        plt.tight_layout()
        plt.savefig(self.opts['save_filepath'])
            
            
        
        
                
        

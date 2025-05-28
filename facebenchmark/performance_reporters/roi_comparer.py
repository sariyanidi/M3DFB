import itertools
import numpy as np

from facebenchmark import BaseReporter
from sklearn.linear_model import LinearRegression

class RoiComparer(BaseReporter):
    
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
        etas = []
        avgR2s = []
        for ec in self.error_computers[mm0][1:]:
            slopes = []
            R2s = []
            
            for ri, rec_method_full in enumerate(self.rec_methods):
                
                ref_errs = []
                est_errs = []
                
                for subj_id in self.subj_ids:
                    ref_errs.append(self.compute_pervertex_error(subj_id, rec_method_full, ref_err_computer).mean())
                    est_errs.append(self.compute_pervertex_error(subj_id, rec_method_full, ec).mean())
                
                mdl = LinearRegression(fit_intercept=False)
                ref_errs = np.array(ref_errs).reshape(-1,1)
                est_errs = np.array(est_errs).reshape(-1,1)
                mdl.fit(ref_errs, est_errs)
                slopes.append(mdl.coef_[0])
                R2s.append(mdl.score(ref_errs, est_errs))
            
            etas.append(100*np.std(slopes)/np.mean(slopes))
            avgR2s.append(np.mean(R2s))

        print(etas)
        print(avgR2s)
            
            
        """
        
            
            
        headers_rec_method = [r.split('/')[1] for r in self.rec_methods]
        headers_ec = [ec.name for ec in self.error_computers[mm0]]
        
        
        Nrec_methods = len(headers_rec_method)
        Nerr_computers = len(headers_ec)
        
        err_table = np.zeros((Nrec_methods, Nerr_computers))
        
        for i, j in itertools.product(range(Nrec_methods), range(Nerr_computers)):
            err_table[i, j] = all_errs[headers_rec_method[i]][headers_ec[j]]
        
        err_table /= self.opts['divide_error_by']
        err_table = err_table.round(self.opts['round_digits'])
        headers_columns = headers_ec
        headers_rows = headers_rec_method
        
        if self.opts['layout'] == "err_computers/rec_methods":
            err_table = err_table.T
            headers_rows  = headers_ec
            headers_columns = headers_rec_method
        
        if self.opts['show_row_headers']:
            err_table = np.concatenate(([headers_columns], err_table))
            headers_rows = [''] + headers_rows
        
        if self.opts['show_col_headers']:
            err_table = np.concatenate((np.array([headers_rows]).reshape(-1,1), err_table), axis=1)
            
        for row in err_table:
            print(self.opts['delimiter'].join(row))
            """
        
        
        
        
        
                
        

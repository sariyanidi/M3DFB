import itertools
import numpy as np

from facebenchmark import BaseReporter

class ResultsTable(BaseReporter):
    
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
        
        all_errs = {}
        
        mm0 = self.rec_methods[0].split('/')[0]
        headers_rec_method = [r.split('/')[1] for r in self.rec_methods]
        headers_ec = [ec.name for ec in self.error_computers[mm0]]
        
        
        for ri, rec_method_full in enumerate(self.rec_methods):
            mm_name = self.rec_methods[0].split('/')[0]
            
            rec_method_name = rec_method_full.split('/')[1]
            
            all_errs[rec_method_name] = {}
            for ec in self.error_computers[mm_name]:
                cerrs = []
                for subj_id in self.subj_ids:
                    ptwise_err = self.compute_pervertex_error(subj_id, rec_method_full, ec)
                    if self.opts["persubj_statistic"] == "mean":
                        cerrs.append(ptwise_err.mean())
                    elif self.opts["persubj_statistic"] == "median":
                        cerrs.append(ptwise_err.median())
            
                if self.opts["overall_statistic"] == "mean":
                    all_errs[rec_method_name][ec.name] = np.mean(cerrs)
                elif self.opts["overall_statistic"] == "median":
                    all_errs[rec_method_name][ec.name] = np.median(cerrs)
        
        Nrec_methods = len(headers_rec_method)
        Nerr_computers = len(headers_ec)
        
        err_table = np.zeros((Nrec_methods, Nerr_computers))
        
        for i, j in itertools.product(range(Nrec_methods), range(Nerr_computers)):
            err_table[i, j] = all_errs[headers_rec_method[i]][headers_ec[j]]
            
        headers_rec_method_ = headers_rec_method
        max_len = max([len(x) for x in headers_rec_method_])
        headers_rec_method = [f'{x:{max_len}}' for x in headers_rec_method_]
        
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
            headers_rows = ['\t'] + headers_rows
        
        if self.opts['show_col_headers']:
            err_table = np.concatenate((np.array([headers_rows]).reshape(-1,1), err_table), axis=1)
            
        for row in err_table:
            print(self.opts['delimiter'].join(row))
        
        
        
        
        
                
        

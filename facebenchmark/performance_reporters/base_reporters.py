import os
import sys
import copy
import time
import itertools
import numpy as np

from glob import glob
from . import ErrorComputer
from facebenchmark import compute_landmark_base_vertex_weights


class PoolParams():
    def __init__(self, reporter, subj_id, rec_method_full, ec):
        self.reporter = reporter
        self.subj_id = subj_id
        self.rec_method_full = rec_method_full
        self.ec = ec


def func(pp):
    pp.reporter.compute_pervertex_error(pp.subj_id, pp.rec_method_full, pp.ec)


class BaseReporter():
    
    def __init__(self, db_info, error_computer_recipes, 
                 rec_methods, mms_info, data_root_dir,
                 opts = {}, use_cache = True, verbosity_level=1,
                 num_processes=1, subj_ids = None):
        
        mm_names = list(mms_info.keys())
        self.opts = opts
        self.db_info = db_info
        self.mms_info = mms_info
        self.use_cache = use_cache
        self.num_processes = num_processes
        self.verbosity_level = verbosity_level
        
        self.vertex_maps = None

        if 'weight_threshold' in self.opts:
            self.vertex_maps = self.compute_vertex_maps_to_report_results_on()
            
        self.error_computers = {
            mm_name: 
            [ErrorComputer(r, mm=self.mms_info[mm_name]) for r in error_computer_recipes]
            for mm_name in mm_names
        }
        self.error_computer_names = [r['name'] for r in error_computer_recipes]
        
        self.rec_methods = rec_methods
        self.DATA_ROOT = data_root_dir
        
        self.Nsubjs = self.count_dataset_subjects()
        
        if 'num_subjects' in self.opts:
            self.Nsubjs = self.opts['num_subjects']
        
        self.subj_ids = self.get_subjs_to_process() if subj_ids is None else subj_ids
    
    
    
    def compute_all_pervertex_errors(self):
        import multiprocessing as mp
        
        param_sets = []
        for ri, rec_method_full in enumerate(self.rec_methods):
            mm_name = rec_method_full.split('/')[0]
            for ec, subj_id in itertools.product(self.error_computers[mm_name], self.subj_ids):
                cache_fpath = self.construct_cache_filepath(subj_id, rec_method_full, ec)
                valid_cache_exists = os.path.exists(cache_fpath)
                
                if not valid_cache_exists:
                    param_sets.append(PoolParams(self, subj_id, rec_method_full, ec))
        
        pool = mp.Pool(processes=self.num_processes)
        pool.map(func, param_sets)
    
    
    def count_dataset_subjects(self):
        Gdir = os.path.join(self.DATA_ROOT, self.db_info['name'], 'Gmeshes')
        return len(glob(f'{Gdir}/*.txt'))
    
    
    def get_subjs_to_process(self):
        
        ids_per_rec_method = {r : set() for r in self.rec_methods}
        for rec_method in self.rec_methods:
            for subj_id in range(self.Nsubjs):
                fp = self.construct_Rfilepath(subj_id, rec_method)
                if os.path.exists(fp):
                    R, Rlmks = self.get_Rdata(subj_id, rec_method)
                    if np.sum(np.isnan(R)) == 0:
                        ids_per_rec_method[rec_method].add(subj_id)
                    
        
        all_subjs = set(list(range(self.Nsubjs)))
        selected_subjs = copy.deepcopy(all_subjs)
        
        for v in ids_per_rec_method.values():
            selected_subjs = selected_subjs.intersection(v)
        
        if len(selected_subjs) < len(all_subjs) and self.verbosity_level > 0:
            print("R meshes are missing or have NaN values for some methods:", file=sys.stderr)
            for rec_method in self.rec_methods:
                gap = len(all_subjs)-len(ids_per_rec_method[rec_method])
                if gap > 0:
                    if gap == 1:
                        print(f"\t{gap} R mesh is missing for {rec_method} method.", file=sys.stderr)
                    else:
                        print(f"\t{gap} R meshes are missing for the {rec_method} method.", file=sys.stderr)
        return selected_subjs
            
    
    def compute_pervertex_error(self, subj_id, rec_method_full, ec):

        cache_fpath = self.construct_cache_filepath(subj_id, rec_method_full, ec)
        valid_cache_exists = os.path.exists(cache_fpath)
        mm_key = rec_method_full.split('/')[0]

        err = None

        if valid_cache_exists:
            err = np.loadtxt(cache_fpath)
            
        if valid_cache_exists and (len(err) != self.mms_info[mm_key]['Npoints'] or np.sum(np.isnan(err)) > 0):
            print('cache is invalid -- ignoring this cache and computing from scratch')
            valid_cache_exists = False

        if not valid_cache_exists or not self.use_cache:
            R, Rlmks = self.get_Rdata(subj_id, rec_method_full)
            G, Glmks = self.get_Gdata(subj_id)
            
            if 'divide_gmeshes_by' in self.opts:
                G /= self.opts['divide_gmeshes_by']
                Glmks /= self.opts['divide_gmeshes_by']
            try:
                err = ec.compute(R, G, Rlmks, Glmks)
            except:
                err = np.nan*np.ones((R.shape[0], 1))
            
            if self.use_cache:
                np.savetxt(cache_fpath, err)
        
        if self.vertex_maps is None:
            return err
        else:
            return err[self.vertex_maps[mm_key]]
    
    
    def compute_pervertex_error_all_computers(self, subj_id, rec_method_full):
        
        mm0 = rec_method_full.split('/')[0]

        errs = {}
        for ec in self.error_computers[mm0]:
            errs[ec.name] = self.compute_pervertex_error(subj_id, rec_method_full, ec)
        
        return errs
    
    
    def construct_cache_filepath(self, subj_id, rec_method_full, ec):
        
        db_name = self.db_info['name']
        mm_name = rec_method_full.split('/')[0]
        rec_method = rec_method_full.split('/')[1]
        mm_version = self.mms_info[mm_name]['version']
        
        cache_fpath = os.path.join(self.DATA_ROOT, 'cache', db_name, 'errs', mm_name, mm_version, rec_method, f'id{subj_id:04d}'+ec.name+ec.key()+'.err')
        cache_dir = os.path.dirname(cache_fpath)
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        return cache_fpath

    
    def construct_Rfilepath(self, subj_id, rec_method_full):
        
        db_name = self.db_info['name']
        mm_name = rec_method_full.split('/')[0]
        rec_method = rec_method_full.split('/')[1]
        mm_version = self.mms_info[mm_name]['version']
                
        return os.path.join(self.DATA_ROOT, db_name, 'Rmeshes', mm_name, mm_version, rec_method, f'id{subj_id:04d}.txt')
        
    

    def get_Rdata(self, subj_id, rec_method_full):
        
        mm_name = rec_method_full.split('/')[0]
        lmk_indices = self.mms_info[mm_name]['lmk_indices']
        Rpath = self.construct_Rfilepath(subj_id, rec_method_full)
        
        R = np.loadtxt(Rpath)
        Rlmks = R[lmk_indices, :]
        
        return R, Rlmks
    
    
    def get_Gdata(self, subj_id):
        bname = os.path.join(self.DATA_ROOT, self.db_info['name'], 'Gmeshes', f'id{subj_id:04d}')
        Gpath = bname + '.txt'
        Glmks_path = bname + '.lmks'
        
        return np.loadtxt(Gpath), np.loadtxt(Glmks_path) 
    
    
    def compute_vertex_maps_to_report_results_on(self):
        import matplotlib.pyplot as plt
        vertex_maps = {}
        
        for mm_key in self.mms_info:
            w = compute_landmark_base_vertex_weights(self.mms_info[mm_key], 'sqrt', 'mixed')
            """
            Xmean = self.mms_info[mm_key]['mean_face_shape']
            w[w<self.opts['weight_threshold']] = 0
            plt.scatter(Xmean[:,0], Xmean[:,1], c=w, cmap='jet')
            """
            vertex_maps[mm_key] = w>self.opts['weight_threshold']
        
        return vertex_maps
        
    
    def produce(self, R, G):
        raise NotImplementedError("""This class is not implemented 
                                  (a virtual method has been called)""")



class MetaEvaluator(BaseReporter):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.db_info['is_synthetic']
        assert len(self.error_computer_names) == 2
        assert self.error_computer_names[0].lower() == 'true'



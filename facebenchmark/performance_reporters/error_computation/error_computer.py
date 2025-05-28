import copy
import time

class ErrorComputer():
    
    def __init__(self, recipe, mm):
        
        self.lmk_indices = mm['lmk_indices']
        
        self.mesh_cropper = None
        self.rigid_aligner = None
        self.nonrigid_aligner = None
        self.corr_establisher = None
        self.corrector = None
        
        self.name = recipe['name']
        
        module = __import__("facebenchmark").performance_reporters.error_computation
        
        if 'mesh_cropper' in recipe and recipe['mesh_cropper']:
            mesh_cropper_class = getattr(module, recipe['mesh_cropper']['type'])
            self.mesh_cropper = mesh_cropper_class(opts=recipe['mesh_cropper']['opts'])
        
        if recipe['rigid_aligner'] is not None:
            rigid_aligner_class = getattr(module, recipe['rigid_aligner']['type'])
            self.rigid_aligner = rigid_aligner_class(opts=recipe['rigid_aligner']['opts'])
        
        if recipe['nonrigid_aligner'] is not None:
            nonrigid_aligner_class = getattr(module, recipe['nonrigid_aligner']['type'])
            self.nonrigid_aligner = nonrigid_aligner_class(opts=recipe['nonrigid_aligner']['opts'])
        
        if recipe['corrector'] is not None:
            corrector_class = getattr(module, recipe['corrector']['type'])
            self.corrector = corrector_class(opts=recipe['corrector']['opts'], mm=mm)
        
        if recipe['corr_establisher'] is not None:
            corr_establisher_class = getattr(module, recipe['corr_establisher']['type'])
            self.corr_establisher = corr_establisher_class(opts=recipe['corr_establisher']['opts'])
        
        distance_computer_class = getattr(module, recipe['distance_computer']['type'])
        self.distance_computer = distance_computer_class(opts=recipe['distance_computer']['opts'])
        
    
    def key(self):
        
        def key_single(opts):
            opts_str = []
            for value in opts.values():
                if isinstance(value, bool):
                    opts_str.append(str(int(value)))
                elif value is None:
                    opts_str.append('N')
                elif isinstance(value, list):
                    opts_str.append(''.join([str(x) for x in value]))
                else:
                    opts_str.append(str(value))
        
            return '-'.join(opts_str)
        
        keys = []
        
        if self.rigid_aligner is not None:
            keys.append(key_single(self.rigid_aligner.opts))
        
        if self.nonrigid_aligner is not None:
            keys.append(key_single(self.nonrigid_aligner.opts))
        
        if self.corr_establisher is not None:
            keys.append(key_single(self.corr_establisher.opts))
        
        if self.distance_computer is not None:
            keys.append(key_single(self.distance_computer.opts))
        
        if self.corrector is not None:
            keys.append(key_single(self.corrector.opts))
            
        return '+'.join(keys)
        
    
    def compute(self, R, G, Rlmks, Glmks):
        
        if self.mesh_cropper is not None:
            G = self.mesh_cropper.crop(G, Glmks)
            
        if self.rigid_aligner is not None:
            R, Rlmks = self.rigid_aligner.align(R, G, Rlmks, Glmks)

        if self.nonrigid_aligner is not None:
            Rref = self.nonrigid_aligner.align(copy.deepcopy(R), G, Glmks, self.lmk_indices)
        else:
            Rref = R
        
        pidx = None
        if self.corr_establisher is not None:
            pidx = self.corr_establisher.establish(Rref, G)

        if self.corrector is not None:
            dG = self.corrector.correct(Rref, G[pidx])
            G[pidx] -= dG
       
        ptwise_error = self.distance_computer.compute(R, G, pidx, Rlmks, Glmks) 
        
        return ptwise_error
    
    
    def plot_RG(self, R, G):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(30,30))
        
        ss = 1
        plt.subplot(2,2,1)
        plt.plot(G[::ss,0], G[::ss,1], '.')
        plt.plot(R[::ss,0], R[::ss,1], '.')
        
        plt.subplot(2,2,2)
        plt.plot(G[::ss,2], G[::ss,1], '.')
        plt.plot(R[::ss,2], R[::ss,1], '.')
        
    
    
        
        


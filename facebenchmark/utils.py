import numpy as np

def compute_landmark_base_vertex_weights(mm, weight_power, weight_strategy):
    
    Xmean = mm['mean_face_shape']
    if isinstance(Xmean, list):
        Xmean = np.array(Xmean)
        
    lix = mm['leye_oc_rel_index']
    rix = mm['reye_oc_rel_index']
    lmk_indices = mm['lmk_indices']
    iod =  np.linalg.norm(Xmean[lmk_indices[lix],:]-Xmean[lmk_indices[rix],:])
    Xmean /= iod
    Xmean_lmks = Xmean[lmk_indices,:]
    
    Nl = len(lmk_indices)
            
    adists = []
    for i in range(Nl):
        dists = np.sqrt(np.sum(((Xmean_lmks[i,:]-Xmean)**2), axis=1))
        adists.append(dists)
        dth = 0.01
        dists[np.where(dists < dth)] = dth # max(dists)
    
    madists = np.array(adists).min(axis=0)#/np.median(np.array(adists),axis=0)
    weights_min = (1./madists)
    
    ref_lis = [0, 2, 4, 5, 7, 9, 20, 21, 23, 24, 26, 27, 29, 30, 19, 22, 25, 28, 13, 
               14, 18, 31, 33, 34, 35, 37, 44, 45, 46, 39, 40, 41, 49, 48, 50]
    adists = []
    if Nl == 51:
        for i in range(len(ref_lis)):
            dists = np.sqrt(np.sum(((Xmean[lmk_indices[ref_lis[i]],:]-Xmean)**2), axis=1))
            adists.append(dists)
    else:
        for i in range(len(lmk_indices)):
            dists = np.sqrt(np.sum(((Xmean[lmk_indices[i],:]-Xmean)**2), axis=1))
            adists.append(dists)
    
    mean_dists = np.mean(np.array(adists),axis=0)#/np.median(np.array(adists),axis=0)
    
    weights_mean = (1./np.abs(mean_dists-0.48))
    
    if weight_strategy == 'mixed':
        weights = (weights_mean+weights_min)/2
    elif weight_strategy == 'min':
        weights = weights_min
    elif weight_strategy == 'mean':
        weights = weights_mean
    
    weights[np.where(weights<1)] = 1

    if weight_power == 'square':
        # weights = (weights**2)/(np.max(weights**2)/np.max(weights))
        weights = (weights**2)/(np.mean(weights**2)/np.mean(weights))
    elif weight_power == 'sqrt':
        weights = np.sqrt(weights)
    
    return weights
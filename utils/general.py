import numpy as np


# ====== general =================

def dict_min_key_by_val(d:dict, ind = 0):
    min_val = np.inf
    
    for key,val in d.items():
        if hasattr(val,'__iter__'):
            val = val[ind]
        if val < min_val:
            min_val = val
            min_key = key
            
    return min_key
    
def sample_box(n,bbox):
    ymax,ymin,xmin,xmax = bbox
    x = np.random.rand(n)*(xmax - xmin) + xmin
    y = np.random.rand(n)*(ymax - ymin) + ymin
    
    return np.stack([x,y],axis = 1)
    
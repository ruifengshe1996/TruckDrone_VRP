# for pandas 
import pandas as pd
import numpy as np

def find_df_max(df:pd.DataFrame):
    
    max_Series = df.idxmax(axis = 1)
    max_val = 0
    
    for ind in max_Series.index:
        
        col = max_Series[ind]
        if df.loc[ind,col] > max_val:
            max_val = df.loc[ind,col]
            max_ind = ind
            max_col = col
    
    return max_val,max_ind,max_col

def find_df_min(df:pd.DataFrame):
    
    min_Series = df.idxmin(axis = 1)
    min_val = np.inf
    
    for ind in min_Series.index:
        
        col = min_Series[ind]
        if df.loc[ind,col] < min_val:
            min_val = df.loc[ind,col]
            min_ind = ind
            min_col = col
    
    return min_val,min_ind,min_col
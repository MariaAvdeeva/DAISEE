import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse

import anndata
import pyliger

from pyliger.factorization._utilities import nnlsm_blockpivot
from pyliger import create_liger

def setup_data(dat = None, 
               bt_design = None):
    samples =  bt_design.index.values
    batch_design = bt_design['batch']
    condition_column = bt_design.columns.values[1]
    treatment_design  = bt_design[condition_column]
    adata_list = []
    for samp in samples:
        cur_dat = dat[dat.obs['sample'] == samp,:].layers['raw'].todense()
        a = anndata.AnnData(cur_dat.copy(), 
                            obs = range(cur_dat.shape[0]), 
                            var = range(cur_dat.shape[1]))
        cur_batch = batch_design[samp]
        cur_stim = treatment_design[samp]
        cur_label = 'Batch '+str(cur_batch)+', Stim '+str(cur_stim)
        a.obs.index.names = ['cell']
        a.uns['sample_name'] = cur_label
        a.var.index.names = ['feature']
        adata_list.append(a)
        #print(a.shape)
    return adata_list

def setup_daisee(adata_list = None, 
                 dat = None,
                 bt_design = None,
                 remove_missing = False,
                 scaled_key = 'scaled_by_sample'):
    # samples = np.unique(dat.obs['sample'])
    if adata_list is None:
        adata_list = setup_data(dat, bt_design)
    samples= bt_design.index.values
    liger_obj = pyliger.create_liger(adata_list, remove_missing = remove_missing)
    from scipy import sparse
    pyliger.normalize(liger_obj)
    #pyliger.select_genes(liger_obj, var_thresh = -0.1, alpha_thresh = 20)
    for (i, samp) in enumerate(samples):  
        #print(samp)
        cur_data = dat[dat.obs['sample'] == samp,:]
        liger_obj.adata_list[i].layers['norm_data']=cur_data.layers['norm']
        #liger_obj.adata_list[i].layers['log1p']=cur_data.layers['log1p']
        liger_obj.adata_list[i].layers['scale_data'] = sparse.csr_matrix(cur_data.layers[scaled_key])
    liger_obj.var_genes = adata_list[0].var_names
    return liger_obj
#pyliger.scale_not_center(liger_obj)


def setup_data_condition(dat = None, 
                         bt_design = None):
    batch_design = bt_design['batch'].values
    condition_column = bt_design.columns.values[1]
    treatment_design  = bt_design[condition_column].values
    adata_list = []
    for tr in np.unique(treatment_design):
        cur_dat = dat[dat.obs[condition_column] == tr,:].layers['raw']
        
        a = anndata.AnnData(cur_dat.copy(), 
                            obs = range(cur_dat.shape[0]), 
                            var = range(cur_dat.shape[1]))
        a = anndata.concat(a)
        a.obs_names_make_unique()
        cur_label = 'Stim '+str(tr)
        a.obs.index.names = ['cell']
        a.uns['sample_name'] = cur_label
        a.var.index.names = ['feature']
        #a.layers['norm_data'] = all_pts[samp]
        adata_list.append(a)
        print(a.shape)
    return adata_list

def setup_liger_condition(adata_list,
                          dat = None, #scaled = True, 
                          bt_design = None, 
                          remove_missing = False,
                          scaled_key = 'scaled_by_sample'):
    liger_obj = pyliger.create_liger(adata_list, 
                                      remove_missing = remove_missing)
    pyliger.normalize(liger_obj)
    condition_column = bt_design.columns.values[1]
    treatment_design  = bt_design[condition_column].values
    liger_obj.var_genes = adata_list[0].var_names
    for (i, tr) in enumerate(np.unique(treatment_design)):  
        cur_dat = dat[dat.obs[condition_column] == tr,:]
        liger_obj.adata_list[i].layers['norm_data']=cur_dat.layers['norm']
        liger_obj.adata_list[i].layers['scale_data'] = sparse.csr_matrix(cur_dat.layers[scaled_key])
    return liger_obj


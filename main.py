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




def optimize_ALS_full_des(liger_object,
                 k,
                 batch_design = None,
                 treatment_design = None,
                 value_lambda_B=5.0,
                 value_lambda_T=5.0,
                 thresh=1e-6,
                 max_iters=30,
                 nrep=1,
                 H_init=None,
                 W_init=None,
                 VB_init=None,
                 VT_init=None,
                force_0 = [],
                                
                 rand_seed=1,
                 print_obj=False):
    """ Perform iNMF on scaled datasets

    Perform integrative non-negative matrix factorization to return factorized H, W, and V matrices.
    It optimizes the iNMF objective function using block coordinate descent (alternating non-negative
    least squares), where the number of factors is set by k. TODO: include objective function equation here in documentation (using deqn)

    For each dataset, this factorization produces an H matrix (cells by k), a V matrix (k by genes),
    and a shared W matrix (k by genes). The H matrices represent the cell factor loadings.
    W is held consistent among all datasets, as it represents the shared components of the metagenes
    across datasets. The V matrices represent the dataset-specific components of the metagenes.

    Args:
        liger_object(liger):
            Should normalize, select genes, and scale before calling.
        k(int):
            Inner dimension of factorization (number of factors). Run suggestK to determine
            appropriate value; a general rule of thumb is that a higher k will be needed for datasets with
            more sub-structure.
        value_lambda(float): optional, (default 5.0)
            Regularization parameter. Larger values penalize dataset-specific effects more
            strongly (ie. alignment should increase as lambda increases). Run suggestLambda to determine
            most appropriate value for balancing dataset alignment and agreement.
        thresh(float): optional, (default 1e-6)
            Convergence threshold. Convergence occurs when |obj0-obj|/(mean(obj0,obj)) < thresh.
        max_iters(int): optional, (default 30)
            Maximum number of block coordinate descent iterations to perform
        nrep(int): optional, (default 1)
            Number of restarts to perform (iNMF objective function is non-convex, so taking the
            best objective from multiple successive initializations is recommended). For easier
            reproducibility, this increments the random seed by 1 for each consecutive restart, so future
            factorizations of the same dataset can be run with one rep if necessary.
        H_init(): optional, (default None)
            Initial values to use for H matrices.
        W_init(): optional, (default None)
            Initial values to use for W matrix.

        V_init(): optional, (default None)
            Initial values to use for V matrices.
        rand_seed(seed): optional, (default 1)
            Random seed to allow reproducible results
        print_obj(bool): optional, (default False)
            Print objective function values after convergence.

    Return:
        liger_object(liger):
            liger object with H, W, and V attributes.

    Usage:
        >>> adata1 = AnnData(np.arange(12).reshape((4, 3)))
        >>> adata2 = AnnData(np.arange(12).reshape((4, 3)))
        >>> ligerex = pyliger.create_liger([adata1, adata2])
        >>> ligerex = pyliger.normalize(ligerex)
        >>> ligerex = pyliger.select_genes(ligerex) # select genes
        >>> ligerex = pyliger.scale_not_center(ligerex)
        >>> ligerex = pyliger.optimize_ALS(ligerex, k = 20, value_lambda = 5, nrep = 3) # get factorization using three restarts and 20 factors
    """
    ### 0. Extract required information
    # prepare basic dataset profiles
    N = liger_object.num_samples  # number of total input hdf5 files
    ns = [adata.shape[0] for adata in liger_object.adata_list]  # number of cells in each hdf5 files
    num_genes = len(liger_object.var_genes)  # number of variable genes
    X = [adata.layers['scale_data'].toarray() for adata in liger_object.adata_list]
    
    B = np.array(batch_design)
    T = np.array(treatment_design)

    if k >= np.min(ns):
        raise ValueError('Select k lower than the number of cells in smallest dataset: {}'.format(np.min(ns)))

    if k >= num_genes:
        raise ValueError('Select k lower than the number of variable genes: {}'.format(len(liger_object.var_genes)))

    best_obj = np.Inf

    for j in range(nrep):
        np.random.seed(seed=rand_seed + j - 1)

        ### 1. Initialization (W, V_i, H_i)
        W = np.abs(np.random.uniform(0, 2, (k, num_genes)))
        VB = [np.abs(np.random.uniform(0, 1, (k, num_genes))) for i in np.unique(B)]
        VT = [np.abs(np.random.uniform(0, 1, (k, num_genes))) for i in np.unique(T)]
        for l in force_0:
            VT[l] = np.zeros((k, num_genes))
        H = [np.abs(np.random.uniform(0, 2, (ns[i], k))) for i in range(N)]

        if W_init is not None:
            W = W_init

        if VB_init is not None:
            VB = VB_init
            
        if VT_init is not None:
            VT = VT_init

        if H_init is not None:
            H = H_init

        delta = 1
        sqrt_lambda_B = np.sqrt(value_lambda_B)
        sqrt_lambda_T = np.sqrt(value_lambda_T)

        # Initial training obj
        obj_train_approximation = 0
        obj_train_penalty_B = 0
        obj_train_penalty_T = 0
        for i in range(N):
            V = VB[B[i]]+VT[T[i]]
            obj_train_approximation += np.linalg.norm(X[i] - H[i] @ (W + V)) ** 2
            obj_train_penalty_B += value_lambda_B*np.linalg.norm(H[i] @ VB[B[i]]) ** 2
            obj_train_penalty_T += value_lambda_T*np.linalg.norm(H[i] @ VT[T[i]]) ** 2

        obj0 = obj_train_approximation + obj_train_penalty_B + obj_train_penalty_T

        ### 2. Iteration starts here
        for iter in tqdm(range(max_iters)):
            if delta > thresh:
                ## 1) update H matrix
                for i in range(N):
                    V = VB[B[i]]+VT[T[i]]
                    H[i] = nnlsm_blockpivot(A=np.hstack(((W + V), sqrt_lambda_B * VB[B[i]], sqrt_lambda_T * VT[T[i]])).transpose(),
                                            B=np.hstack((X[i], np.zeros((ns[i], num_genes)), np.zeros((ns[i], num_genes)))).transpose())[0].transpose()

                ## 2) update V matrix
                for l in np.unique(B):
                    cur_samples = np.where(B == l)[0]
                    #print(cur_samples)
                    VB[l] = nnlsm_blockpivot(A=np.vstack([np.vstack((H[i], sqrt_lambda_B * H[i])) for i in cur_samples]),
                                                B=np.vstack([np.vstack(((X[i] - H[i] @ W - H[i] @ VT[T[i]]), np.zeros((ns[i], num_genes)))) for i in cur_samples]))[0]
                for l in np.unique(T):
                    if l in force_0:
                        continue
                    else:
                        cur_samples = np.where(T == l)[0]
                        #print(cur_samples)
                        VT[l] = nnlsm_blockpivot(A=np.vstack([np.vstack((H[i], sqrt_lambda_T * H[i])) for i in cur_samples]),
                                                    B=np.vstack([np.vstack(((X[i] - H[i] @ W - H[i] @ VB[B[i]]), np.zeros((ns[i], num_genes)))) for i in cur_samples]))[0]

                ## 3) update W matrix
                W = nnlsm_blockpivot(A=np.vstack(H), B=np.vstack([(X[i] - H[i] @ (VB[B[i]]+VT[T[i]])) for i in range(N)]))[0]

                obj_train_prev = obj0
                obj_train_approximation = 0
                obj_train_penalty_B = 0
                obj_train_penalty_T = 0
                for i in range(N):
                    V = VB[B[i]]+VT[T[i]]
                    obj_train_approximation += np.linalg.norm(X[i] - H[i] @ (W + V)) ** 2
                    obj_train_penalty_B += value_lambda_B*np.linalg.norm(H[i] @ VB[B[i]]) ** 2
                    obj_train_penalty_T += value_lambda_T*np.linalg.norm(H[i] @ VT[T[i]]) ** 2
                obj0 = obj_train_approximation + obj_train_penalty_B+obj_train_penalty_T
                #print(obj0)
                delta = np.absolute(obj_train_prev - obj0) / ((obj_train_prev + obj0) / 2)
            else:
                continue

        if obj0 < best_obj:
            final_W = W
            final_H = H
            final_VB = VB
            final_VT = VT
            best_obj = obj0
            best_seed = rand_seed + i - 1

        if print_obj:
            print('Objective: {}'.format(best_obj))
    
    #liger_object.W = final_W.transpose()
    ### 3. Save results into the liger_object
    #create copy
    new_object = create_liger(liger_object.adata_list, remove_missing = False)
    for i in range(N):
        new_object.adata_list[i].obsm['H'] = final_H[i]
        new_object.adata_list[i].varm['W'] = final_W.transpose()
        new_object.adata_list[i].varm['VB'] = final_VB[B[i]].transpose()
        new_object.adata_list[i].varm['VT'] = final_VT[T[i]].transpose()
        new_object.adata_list[i].varm['V'] = (final_VB[B[i]]+final_VT[T[i]]).transpose()
        #idx = liger_object.adata_list[i].uns['var_gene_idx']
        #shape = liger_object.adata_list[i].shape
        #save_W = np.zeros((shape[1], k))
        #save_W[idx, :] = final_W.transpose()
        #save_V = np.zeros((shape[1], k))
        #save_V[idx, :] = final_V[i].transpose()
        #liger_object.adata_list[i].obsm['H'] = final_H[i]
        #liger_object.adata_list[i].varm['W'] = save_W
        #liger_object.adata_list[i].varm['V'] = save_V

    return new_object


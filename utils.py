import numpy as np
import pandas as pd

import anndata

import pyliger
from DAISEE import *

from sklearn.decomposition import NMF, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jaccard
from sklearn.metrics.pairwise import pairwise_distances 

from joblib import Parallel, delayed
import time
from functools import reduce

from scipy import sparse

import matplotlib.pyplot as plt

def test_B_lambda(val_B, rand_seed, 
                  dat = None, bt_design = None,
                  max_iters = 100, 
                  nrep = 3, 
                  k = 30, 
                  min_cells = None,
                  n_jobs = -1,
                  k_agree = 100,
                  calc_agreement = False,
                  ref_dataset = None):
    al = {}
    ag = {}
    for rep in range(nrep):
        batch_design = bt_design['batch']
        condition_column = bt_design.columns.values[1]
        treatment_design  = bt_design[condition_column]
        samples =  bt_design.index.values
        
        daisee_obj = setup_daisee(adata_list = None, 
                 dat = dat,
                 bt_design = bt_design)
        daisee_obj = optimize_ALS_full_des(daisee_obj, 
                                           batch_design = batch_design,
                                           treatment_design = treatment_design,
                                           value_lambda_B=val_B,
                                           value_lambda_T=5,
                                           force_0 = [],
                                           nrep = 1, 
                                           rand_seed=rand_seed+rep,
                                           k = k,
                                           max_iters = max_iters)
        
        ind = np.concatenate([np.where(dat.obs['sample'] == samp)[0] for samp in samples])
        pyliger.quantile_norm(daisee_obj, 
                          #quantiles = 50,
                          ref_dataset =ref_dataset)
        H_norm = np.vstack([ada.obsm['H_norm'] for ada in daisee_obj.adata_list])
        H_norm = pd.DataFrame(H_norm)
        H_norm.index = dat[ind,:].obs.index
        print('Testing '+str(val_B)+'...')
        al[rep] = calcAlignment(daisee_obj, 
                                H_norm,
                                dat[ind,:].obs['batch'], 
                                n_jobs = n_jobs,
                                rand_seed=rand_seed+rep, 
                                min_cells = min_cells,
                                by_cell = False)
        if calc_agreement:
            ag[rep] = calcAgreement(daisee_obj, 
                                H_norm, 
                                dat[ind,:], 
                                samples,
                                ndims=k, 
                                k=k_agree, 
                                use_aligned=True, 
                                rand_seed=rand_seed+rep, 
                                n_jobs = n_jobs, 
                                by_dataset=False)
    if calc_agreement:
        return al, ag
    else:
        return al


def kl_divergence_uniform(object, Hs=None):
    if Hs is None:
        Hs = object.H

    n_cells = sum([H.shape[0] for H in Hs])
    n_factors = Hs[0].shape[1]
    dataset_list = []
    #print(len(Hs))
    for H in Hs:
        scaled = H / ((np.linalg.norm(H, axis=0))/np.sqrt(H.shape[0]-1))

        inflated = np.where(scaled == 0, 1e-20, scaled)
        inflated /= inflated.sum(axis=1, keepdims=True)
        divs = np.apply_along_axis(lambda x: np.log2(n_factors) + np.sum(np.log2(x) * x), axis=1, arr=inflated)
        #print(divs)
        dataset_list.append(divs)

    return dataset_list


def worker(i, k_test = range(6, 41, 2), 
           dat = None,
           bt_design = None,
           value_lambda_B = 5, 
           value_lambda_T = 5, 
           thresh = 1e-4, 
           max_iters = 30, 
           rand_seed = 2):
    #if i != len(k_test) - 1:
    batch_design = bt_design['batch']
    condition_column = bt_design.columns.values[1]
    treatment_design  = bt_design[condition_column]
    daisee_obj = setup_daisee(adata_list = None, 
                 dat = dat,
                 bt_design = bt_design)
    ob_test = optimize_ALS_full_des(daisee_obj, k = k_test[i], 
                                              batch_design = batch_design, 
                                              treatment_design = treatment_design,
                                              value_lambda_B = value_lambda_B, 
                                              value_lambda_T = value_lambda_T,
                                              thresh = thresh, 
                                              max_iters = max_iters, 
                                              nrep=1, 
                                              rand_seed=rand_seed)
    # else:
    #     ob_test = object
    dataset_split = kl_divergence_uniform(ob_test)
    return np.concatenate(dataset_split)

def suggestK(dat = None,
             bt_design = None,
             k_test = range(6, 41, 2), 
             value_lambda_B = 5, 
             value_lambda_T = 5,
             thresh=1e-4, max_iters=30, 
             num_cores=1,
             rand_seed=2, 
             nrep=1, 
             plot_log2=False, 
             return_data=False, 
             return_raw=False, 
             verbose=True):
    
    time_start = time.time()
    if verbose:
        print("This operation may take several minutes depending on the number of values being tested")
    
    rep_data = []
    for r in range(nrep):
        
        if verbose:
            print('Testing different choices of k')
        
        results = Parallel(n_jobs=num_cores)(delayed(worker)(i, 
                                                             k_test, 
                                                             dat, bt_design,
                                                           value_lambda_B, value_lambda_T,
                                                           thresh, max_iters, 
                                                            rand_seed + r - 1) for i in range(len(k_test)))
        data_matrix = np.vstack(results)
        rep_data.append(data_matrix)
    
    medians = np.column_stack([np.median(data, axis=1) for data in rep_data])
    print(medians)
    if len(medians.shape) == 1:
        medians = medians.reshape(-1,1)
    
    mean_kls = np.mean(medians, axis=1)
    
    time_elapsed = time.time() - time_start
    if verbose:
        print(f"\nCompleted in: {time_elapsed} seconds")
    
    df_kl = pd.DataFrame({
        'median_kl': np.concatenate((mean_kls, np.log2(k_test))),
        'k': np.tile(k_test, 2),
        'calc': np.repeat(['KL_div', 'log2(k)'], len(k_test))
    })
    
    if not plot_log2:
        df_kl = df_kl[df_kl['calc'] == 'KL_div']
    
    plt.figure(figsize=(10, 6))
    p1 = plt.plot(df_kl['k'], df_kl['median_kl'], label='Median KL Divergence', marker='o')
    plt.xlabel('K')
    plt.ylabel('Median KL Divergence (across all cells)')
    plt.legend()
    
    if return_data:
        plt.show()
        if return_raw:
            rep_data = [pd.DataFrame(data, index=k_test) for data in rep_data]
            return df_kl, rep_data
        return df_kl
    
    return p1


def calcAgreement(object, H_norm, cur, samples,dr_method="NMF", 
                  ndims=40, 
                  n_jobs = -1,
                  k=15, 
                  use_aligned=True, 
                  rand_seed=42, 
                  by_dataset=False):

    print(f"Reducing dimensionality using {dr_method}")
    np.random.seed(rand_seed)
    dr = []

    if dr_method == "NMF":
        #for data in object.sample_data if isinstance(object.raw_data[0], h5py.File) else object.scale_data:
        for samp in samples:#np.unique(cur.obs['sample']):
            nmf = NMF(n_components=ndims, random_state=rand_seed)
            dr.append(nmf.fit_transform(cur[cur.obs['sample'] == samp,:].layers['scaled_by_sample']))
    print('Transformed')
    ns = [dr_i.shape[0] for dr_i in dr]
    n = sum(ns)
    print(ns)
    jaccard_inds = []
    distorts = []

    for i, dr_i in enumerate(dr):
        jaccard_inds_i = []

        if use_aligned:
            original = H_norm.loc[cur[cur.obs['sample'] == samples[i],:].obs.index,:]
        # else:
        #     original = object.H[i]

        fnn_1 = NearestNeighbors(n_neighbors=k, n_jobs = n_jobs).fit(dr_i)
        fnn_2 = NearestNeighbors(n_neighbors=k, n_jobs = n_jobs).fit(original.values)
        
        neigh1 = np.asarray(fnn_1.kneighbors_graph(dr_i).todense())#.astype(bool)
        neigh2 = np.asarray(fnn_2.kneighbors_graph(original.values).todense())#.astype(bool)
        jaccard_inds_i = np.diag(pairwise_distances(neigh1, neigh2, metric="jaccard", n_jobs=-1))

        # for j in range(ns[i]):
        #     intersect = np.intersect1d(fnn_1.kneighbors([dr_i[j,:]], return_distance=False)[0],
        #                                fnn_2.kneighbors([original.iloc[j,:].values], return_distance=False)[0])
        #     union = np.union1d(fnn_1.kneighbors([dr_i[j,:]], return_distance=False)[0],
        #                        fnn_2.kneighbors([original.iloc[j,:].values], return_distance=False)[0])
        #     jaccard_inds_i.append(len(intersect) / len(union))

        jaccard_inds_i = [val for val in jaccard_inds_i if np.isfinite(val)]
        jaccard_inds.extend(jaccard_inds_i)
        distorts.append(1-np.mean(jaccard_inds_i))#np.mean(jaccard_inds_i))

    if by_dataset:
        return distorts
    return 1-np.mean(jaccard_inds)#np.mean(jaccard_inds)

def calcAlignment(object, H_norm, dataset, k=None, 
                  min_cells = None,
                  rand_seed=1, n_jobs = -1,cells_use=None, cells_comp=None,
                  clusters_use=None, by_cell=False, by_dataset=False):
    if cells_use is None:
        cells_use = H_norm.index.tolist()

    if clusters_use is not None:
        cells_use = [cell for cell in cells_use if object.clusters[cell] in clusters_use]

    if cells_comp is not None:
        nmf_factors = H_norm.loc[cells_use + cells_comp]
        num_cells = len(cells_use + cells_comp)
        func_H = {'cells1': nmf_factors.loc[cells_use],
                  'cells2': nmf_factors.loc[cells_comp]}
        print('Using designated sets cells_use and cells_comp as subsets to compare')
    else:
        nmf_factors = H_norm.loc[cells_use]
        num_cells = len(cells_use)
        func_H = {}
        for x in np.unique(dataset):
            cells_overlap = list(set(cells_use) & set(H_norm.loc[dataset == x,:].index))
            #print(cells_overlap)
            if len(cells_overlap) > 0:
                func_H[x] = H_norm.loc[cells_overlap,:]
            else:
                print(f"Selected subset eliminates dataset {list(object.H.keys())[x]}")
                continue

    num_factors = nmf_factors.shape[1]
    N = len(func_H)

    if N == 1:
        print("Alignment null for single dataset")

    np.random.seed(rand_seed)
    if min_cells is None:
        min_cells = min([len(x) for x in func_H.values()])

    sampled_cells = np.concatenate([np.random.choice(list(func_H[x].index), min_cells) for x in func_H.keys()])

    max_k = len(sampled_cells) - 1

    if k is None:
        k = min(max(int(np.floor(0.01 * num_cells)), 10), max_k)

    elif k > max_k:
        raise ValueError(f"Please select k <= {max_k}")

    knn_graph = NearestNeighbors(n_neighbors=k, n_jobs = n_jobs).fit(nmf_factors.loc[sampled_cells])


    num_sampled = N * min_cells
    num_same_dataset = np.full(num_sampled, k)
    
    alignment_per_cell = np.empty(num_sampled)
    print(k)
    print(num_sampled)
    rel_dat = np.ravel(dataset.loc[sampled_cells])

    neigh = knn_graph.kneighbors(nmf_factors.loc[sampled_cells], return_distance = False)

    batches = dataset.loc[sampled_cells[np.ravel(neigh)]].values.reshape(neigh.shape)
    rep_cells = np.tile(dataset.loc[sampled_cells], (neigh.shape[1],1)).T
    num_same_dataset = (batches == rep_cells).sum(1)
    
    alignment_per_cell = 1 - (num_same_dataset - (k / N)) / (k - k / N)
    if by_dataset:
        alignments = []
        for i in range(1, N + 1):
            start = (i - 1) * min_cells
            end = i * min_cells
            alignment = np.mean(alignment_per_cell[start:end])
            alignments.append(alignment)
        return alignments
    elif by_cell:
        alignment_per_cell_dict = dict(zip(sampled_cells, alignment_per_cell))
        return alignment_per_cell_dict

    return np.mean(alignment_per_cell)

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

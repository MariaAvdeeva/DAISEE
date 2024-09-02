import numpy as np
import pandas as pd
import scanpy as sc


def plot_VT(factor, 
            dat = None,
            VT_df = None, 
            aliases = None,
            cluster_obs = None,
            stim_obs = None,
            to_plot_cond = None,
            num_genes_top = 100,
            num_genes_show = 30,
            lfc = 0.3, 
            pvc = 0.05, 
            fs = 10, 
            ax = None): 
    map_stim_to_index = {}
    samps_and_stim = dat.obs[['sample', stim_obs]].drop_duplicates()
    samples = samps_and_stim['sample'].values
    for (i,x) in enumerate(samps_and_stim[stim_obs].values):
        if x not in map_stim_to_index.keys():
            map_stim_to_index[x] = [i]
        else:
            map_stim_to_index[x].append(i)
    if ax is None:
        ax = plt.gca()
    dataset = map_stim_to_index[to_plot_cond]
    factor_num = np.where(aliases == factor)[0][0]
    cl = dat[dat.obs[cluster_obs] == factor,:].copy()
    sc.tl.rank_genes_groups(cl, groupby = stim_obs,#reference = 'wt',
                            layer = 'log1p',method = 'wilcoxon')
    pvs = sc.get.rank_genes_groups_df(cl, group = None).query('group == "'+to_plot_cond+'"')
    over = pvs.query('logfoldchanges>'+str(lfc)).query('pvals_adj<'+str(pvc)).sort_values('pvals_adj')['names'].values#[:100]
    #under = pvs.query('logfoldchanges<-'+str(lfc)).query('pvals_adj<'+str(pvc)).sort_values('pvals_adj')['names'].values#[:100]
    V = VT_df[samples[dataset[0]]].iloc[:, factor_num].sort_values(ascending = False).iloc[:num_genes_top]
    inters = V.index.intersection(pd.Index(over))
    #print(len(inters))
    sortd = V.loc[inters].iloc[:num_genes_show]#.iloc[::-1]
    sortd = sortd.loc[sortd!=0]
                      
    gene_df = pd.DataFrame(np.diag(sortd), index = sortd.index,
                                                       columns = sortd.index).iloc[::-1,::-1]
    gene_df.plot(kind = 'barh', stacked = True,
                 ax=ax,
                                        cmap = 'plasma', 
                                        width = 0.8,
                                       legend = False,
                                       title = to_plot_cond,
                                       fontsize = fs)
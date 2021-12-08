"""
modified from https://github.com/mims-harvard/nimfa-ipynb/blob/master/ICGC%20and%20Nimfa.ipynb
"""

import itertools

import networkx as nx
import numpy as np
import pandas as pd
import nimfa
from sklearn import preprocessing
from matplotlib import gridspec, pyplot as plt
import scipy.cluster.hierarchy as sch


def rank_estimate(rank_candidates: list, ims):
    """
    estimate rank for NMF using Nimfa package. Sparse non-negative matrix factorization are used here, and the sparseness is imposed on the right matrix (coefficients) The result will be plotted.

    https://github.com/mims-harvard/nimfa-ipynb/blob/master/ICGC%20and%20Nimfa.ipynb

    Parameters:
    --------
        rank_candidates: a list object with ranks to be estimated (integer)

        ims: an array with ion images
    """

    snmf = nimfa.Snmf(ims, seed='random_vcol', max_iter=100)

    summary = snmf.estimate_rank(rank_range=rank_candidates, n_run=10, what='all')

    rss = [summary[rank]['rss'] for rank in rank_cands]

    rss = rss / np.sum(rss)

    coph = [summary[rank]['cophenetic'] for rank in rank_cands]

    disp = [summary[rank]['dispersion'] for rank in rank_cands]

    spar = [summary[rank]['sparseness'] for rank in rank_cands]

    spar_w, spar_h = zip(*spar)

    evar = [summary[rank]['evar'] for rank in rank_cands]

    plt.plot(rank_cands, rss, 'o-', label='RSS', linewidth=2)

    plt.plot(rank_cands, coph, 'o-', label='Cophenetic correlation', linewidth=2)

    plt.plot(rank_cands, disp, 'o-', label='Dispersion', linewidth=2)

    plt.plot(rank_cands, spar_w, 'o-', label='Sparsity (Basis)', linewidth=2)

    plt.plot(rank_cands, spar_h, 'o-', label='Sparsity (Mixture)', linewidth=2)

    plt.plot(rank_cands, evar, 'o-', label='Explained variance', linewidth=2)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, numpoints=1)

    plt.tight_layout()

    plt.show()


def nmf(ims, feature_table, rank: int, n_run=1):

    """
    run nmf with the most stable rank suggested by rank_estimate. set n_run > 1 to get a molecular network based on the co-localization info

    """

    if n_run == 1:

        snmf = nimfa.Snmf(ims, rank=rank, seed='random_vcol', version='r', n_run=n_run, max_iter=500)

    elif n_run > 1:

        snmf = nimfa.Snmf(ims, rank=rank, seed='random_vcol', version='r', track_factor=True, n_run=n_run, max_iter=500)

    snmf_fitted = snmf()

    mzs = list(feature_table.drop(columns = ['x', 'y']).columns())

    basis = pd.DataFrame(snmf_fitted.fit.basis())

    components = pd.DataFrame(snmf_fitted.fit.coef().T, index=mzs)

    if n_run > 1:

        consensus = pd.DataFrame(snmf_fitted.fit.consensus(), index=mzs, columns=mzs)

        consensus.values[[np.arange(len(consensus))] * 2] = np.nan

        consensus = consensus.stack().reset_index()

        consensus = consensus[consensus[0] >= 0.9]

        G = nx.from_pandas_edgelist(consensus, source='level_0', target='level_1')

        return basis, components, G

    else:

        return basis, components



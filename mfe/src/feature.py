"""
modified from https://github.com/mims-harvard/nimfa-ipynb/blob/master/ICGC%20and%20Nimfa.ipynb
"""

import networkx as nx
import nimfa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def rank_estimate(rank_candidates: list, ims):
    """
    estimate rank for NMF using Nimfa package. Sparse non-negative matrix factorization are used here, and the sparseness is imposed on the right matrix (coefficients) The result will be plotted.

    https://github.com/mims-harvard/nimfa-ipynb/blob/master/ICGC%20and%20Nimfa.ipynb

    Parameters:
    --------
        rank_candidates: a list object with ranks to be estimated (integer)

        ims: an array with ion images
    """

    ims = ims.reshape(ims.shape[0], -1)

    snmf = nimfa.Snmf(ims.T, seed='random_vcol', max_iter=100)

    summary = snmf.estimate_rank(rank_range=rank_candidates, n_run=10, what='all')

    rss = [summary[rank]['rss'] for rank in rank_candidates]

    rss = rss / np.sum(rss)

    coph = [summary[rank]['cophenetic'] for rank in rank_candidates]

    disp = [summary[rank]['dispersion'] for rank in rank_candidates]

    spar = [summary[rank]['sparseness'] for rank in rank_candidates]

    spar_w, spar_h = zip(*spar)

    evar = [summary[rank]['evar'] for rank in rank_candidates]

    plt.plot(rank_candidates, rss, 'o-', label='RSS', linewidth=2)

    plt.plot(rank_candidates, coph, 'o-', label='Cophenetic correlation', linewidth=2)

    plt.plot(rank_candidates, disp, 'o-', label='Dispersion', linewidth=2)

    plt.plot(rank_candidates, spar_w, 'o-', label='Sparsity (Basis)', linewidth=2)

    plt.plot(rank_candidates, spar_h, 'o-', label='Sparsity (Mixture)', linewidth=2)

    plt.plot(rank_candidates, evar, 'o-', label='Explained variance', linewidth=2)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, numpoints=1)

    plt.tight_layout()

    plt.show()


def nmf(ims, feature_table, rank: int, n_run=1):
    """
    run nmf with the most stable rank suggested by rank_estimate. set n_run > 1 to get a molecular network based on the co-localization info

    """

    ims = ims.reshape(ims.shape[0], -1).T

    if n_run == 1:

        snmf = nimfa.Snmf(ims, rank=rank, seed='random_vcol', version='r', n_run=n_run, max_iter=500)

    elif n_run > 1:

        snmf = nimfa.Snmf(ims, rank=rank, seed='random_vcol', version='r', track_factor=True, n_run=n_run, max_iter=500)

    snmf_fitted = snmf()

    mzs = list(feature_table.drop(columns=['x', 'y']).columns)

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




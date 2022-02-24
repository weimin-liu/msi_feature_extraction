"""
https://github.com/yal054/snATAC/blob/e70087fb3c71a34dd5a36b72515379115e804c40/snATAC.nmf.py
"""
import itertools
import math

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt, gridspec
from sklearn.decomposition import NMF
import numpy as np
import tqdm


def predict_H(H):
    colmax = np.amax(H, axis=0)
    colsum = np.sum(H, axis=0)
    p = colmax / colsum
    idx = H.argmax(axis=0)
    out = [idx, p]
    return out


def cal_conectivity(H, idx):
    connectivity_mat = np.zeros((H.shape[1], H.shape[1]))
    classN = H.shape[0]
    for i in range(classN):
        xidx = list(np.concatenate(np.where(idx == i)))
        iterables = [xidx, xidx]
        for t in itertools.product(*iterables):
            connectivity_mat[t[0], t[1]] = 1
    return connectivity_mat


def cal_sparsity(X):
    vec = list(np.concatenate(X))
    absSum = np.sum(np.abs(vec))
    n = np.prod(X.shape)
    squareSum = np.sum(np.square(vec))
    numerator = np.sqrt(n) - (absSum / np.sqrt(squareSum))
    denominator = np.sqrt(n) - 1
    sparsity = numerator / denominator
    return sparsity


def cal_rss_mse(W, H, V):
    residualSquare = np.square(W.dot(H) - V)
    rss = np.sum(residualSquare)
    mse = np.mean(residualSquare)
    out = [rss, mse]
    return out


def cal_evar(rss, V):
    evar = 1 - (rss / np.sum(V ** 2))
    return evar


def cal_cophenetic(C):
    X = C
    Z = sch.linkage(X)
    orign_dists = ssd.pdist(X)  # Matrix of original distances between observations
    cophe_dists = sch.cophenet(Z)  # Matrix of cophenetic distances between observations
    corr_coef = np.corrcoef(orign_dists, cophe_dists)[0, 1]
    return corr_coef


def cal_dispersion(C):
    n = C.shape[1]
    corr_disp = np.sum(4 * np.square(np.concatenate(C - 1 / 2))) / (np.square(n))
    return corr_disp


def cal_featureScore_kim(W):
    k = W.shape[1]
    m = W.shape[0]
    s_list = []
    for i in range(m):
        rowsum = np.sum(W[i,])
        p_iq_x_list = []
        for q in range(k):
            p_iq = W[i, q] / rowsum
            if p_iq != 0:
                tmp = p_iq * math.log(p_iq, 2)
            else:
                tmp = 0
            p_iq_x_list.append(tmp)
        s = 1 + 1 / math.log(k, 2) * np.sum(p_iq_x_list)
        s_list.append(s)
    return s_list


def repeated_nmf(V, rank, n_run, *args, **kwargs):
    """

    Args:
        W:
        rank:
        n:

    Returns:

    """
    out_list = []

    consensus = np.zeros((V.shape[1], V.shape[1]))

    for i in tqdm.tqdm(range(n_run)):
        model = NMF(n_components=rank, random_state=i, *args, **kwargs)

        W = model.fit_transform(V)

        H = model.components_

        consensus += cal_conectivity(H, predict_H(H)[0])

        o_sparseH = cal_sparsity(H)

        o_sparseW = cal_sparsity(W)

        o_rss_mse = cal_rss_mse(W, H, V)

        o_rss = o_rss_mse[0]

        o_mse = o_rss_mse[1]

        o_evar = cal_evar(o_rss, V)

        out = [o_sparseH, o_sparseW, o_rss, o_mse, o_evar]

        out_list.append(out)

    out_list = np.array(out_list)

    mean_out = list(np.mean(out_list, axis=0))

    consensus /= n_run

    o_cophcor = cal_cophenetic(consensus)

    o_disp = cal_dispersion(consensus)

    o_fsW = cal_featureScore_kim(W)

    mean_out.append(o_cophcor)

    mean_out.append(o_disp)

    mean_out.append(o_fsW)

    mean_out.append(consensus)

    result_name = ['sparseH', 'sparseW', 'rss', 'mse', 'evar', 'cophcor', 'disp', 'fsW', 'consensus']

    summary = {
        'rank': rank
    }

    for i in range(len(mean_out)):
        summary[result_name[i]] = mean_out[i]

    return summary


def clean_axis(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def save_consensus(C, prefix):

    r_C = 1 - C

    fig = plt.figure(figsize=(13.9, 10))

    heatmapGS = gridspec.GridSpec(1, 2, wspace=.01, hspace=0., width_ratios=[0.25, 1])

    Y = sch.linkage(r_C, method='average')

    denAX = fig.add_subplot(heatmapGS[0, 0])

    denD = sch.dendrogram(Y, orientation='left', link_color_func=lambda k: 'black')

    clean_axis(denAX)

    heatmapAX = fig.add_subplot(heatmapGS[0, 1])

    D = r_C[denD['leaves'], :][:, denD['leaves']]

    axi = heatmapAX.imshow(D, interpolation='nearest', aspect='equal', origin='lower', cmap='RdBu')

    clean_axis(heatmapAX)

    cb = fig.colorbar(axi, fraction=0.046, pad=0.04, aspect=10)

    cb.set_label('Distance', fontsize=20)

    plt.savefig(f'{prefix}_consensus.svg', format='svg')

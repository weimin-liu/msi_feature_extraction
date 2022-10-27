"""
https://github.com/yal054/snATAC/blob/e70087fb3c71a34dd5a36b72515379115e804c40/snATAC.nmf.py
"""
import itertools
from collections import namedtuple
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt, gridspec
from sklearn.decomposition import NMF
import numpy as np
import tqdm
from scipy.stats.stats import pearsonr


def predict_h(matrix_h):
    """
    Args:
        matrix_h:

    Returns:

    """
    col_max = np.amax(matrix_h, axis=0)
    col_sum = np.sum(matrix_h, axis=0)
    p_val = col_max / col_sum
    idx = matrix_h.argmax(axis=0)
    out = [idx, p_val]
    return out


def cal_connectivity(matrix_h, idx):
    """

    Args:
        matrix_h:
        idx:

    Returns:

    """
    connectivity_mat = np.zeros((matrix_h.shape[1], matrix_h.shape[1]))
    class_n = matrix_h.shape[0]
    for i in range(class_n):
        x_idx = list(np.concatenate(np.where(idx == i)))
        iterables = [x_idx, x_idx]
        for t_val in itertools.product(*iterables):
            connectivity_mat[t_val[0], t_val[1]] = 1
    return connectivity_mat


def cal_sparsity(matrix_x):
    """

    Args:
        matrix_x:

    Returns:

    """
    vec = list(np.concatenate(matrix_x))
    abs_sum = np.sum(np.abs(vec))
    n_val = np.prod(matrix_x.shape)
    square_sum = np.sum(np.square(vec))
    numerator = np.sqrt(n_val) - (abs_sum / np.sqrt(square_sum))
    denominator = np.sqrt(n_val) - 1
    sparsity = numerator / denominator
    return sparsity


def cal_rss(matrix_w, matrix_h, matrix_v):
    """

    Args:
        matrix_w:
        matrix_h:
        matrix_v:

    Returns:

    """
    rss = np.sum(np.square(matrix_w.dot(matrix_h) - matrix_v))
    return rss


def cal_mse(matrix_w, matrix_h, matrix_v):
    """

    Args:
        matrix_w:
        matrix_h:
        matrix_v:

    Returns:

    """
    mse = np.mean(np.square(matrix_w.dot(matrix_h) - matrix_v))
    return mse


def cal_evar(rss, matrix_v):
    """

    Args:
        rss:
        matrix_v:

    Returns:

    """
    evar = 1 - (rss / np.sum(matrix_v ** 2))
    return evar


def cal_cophenetic(matrix_consensus):
    """
    Calculate the cophenetic correlation coefficient of consensus matrix
    Args:
        matrix_consensus:

    Returns:

    """
    z_link = sch.linkage(matrix_consensus)
    orign_dists = ssd.pdist(matrix_consensus)  # Matrix of original distances between observations
    cophe_dists = sch.cophenet(z_link)  # Matrix of cophenetic distances between observations
    corr_coef = np.corrcoef(orign_dists, cophe_dists)[0, 1]
    return corr_coef


def cal_dispersion(matrix_consensus):
    """
    Calculate the dispersion correlation coefficient of consensus matrix
    Args:
        matrix_consensus:

    Returns:

    """
    n_col = matrix_consensus.shape[1]
    corr_disp = np.sum(4 * np.square(np.concatenate(matrix_consensus - 1 / 2))) / (np.square(n_col))
    return corr_disp


def accum_w_h(accum_w, accum_h, matrix_w, matrix_h, accum_time):
    """

    Args:
        accum_time:
        accum_w:
        accum_h:
        matrix_w:
        matrix_h:

    Returns:

    """
    tol = 0.9
    corr = np.zeros([accum_w.shape[1], matrix_w.shape[1]])
    for col_m, col_n in itertools.product(range(accum_w.shape[1]), range(matrix_w.shape[1])):
        r_val, _ = pearsonr(accum_w[:, col_m], matrix_w[:, col_n])
        corr[col_m, col_n] = r_val
    for i in range(matrix_w.shape[1]):
        j = np.argmax(corr[:, i])
        if corr[j, i] >= tol:
            accum_time[j] = accum_time[j] + 1
            accum_w[:, j] = accum_w[:, j] + matrix_w[:, i]
            accum_h[j, :] = accum_h[j, :] + matrix_h[i, :]
        else:
            accum_time.append(0)
            accum_w = np.c_[accum_w, matrix_w[:, i]]
            accum_h = np.c_[accum_h.T, matrix_h.T[:, i]].T

    return accum_w, accum_h, accum_time


def repeated_nmf(matrix_v, rank, n_run, *args, **kwargs):
    """
    run NMF repeatedly and find reproducible components
    Args:
        matrix_v:
        rank:
        n_run:
        *args:
        **kwargs:

    Returns:

    """
    out_list = []

    accum_time = []

    Summary = namedtuple('Summary', 'rank sparseH sparseW rss mse evar cophcor disp consensus '
                                    'matrix_w_accum matrix_h_accum accum_time')

    for i in range(rank):
        accum_time.append(0)

    consensus = np.zeros((matrix_v.shape[1], matrix_v.shape[1]))

    matrix_w_accum = 0

    matrix_h_accum = 0

    for i in tqdm.tqdm(range(n_run)):
        model = NMF(n_components=rank, random_state=i, *args, **kwargs)

        matrix_w = model.fit_transform(matrix_v)

        if isinstance(matrix_w_accum, int):
            matrix_w_accum = matrix_w
            matrix_h_accum = model.components_
        elif isinstance(matrix_w_accum, np.ndarray):
            matrix_w_accum, matrix_h_accum, accum_time = accum_w_h(matrix_w_accum, matrix_h_accum,
                                                                   matrix_w, model.components_,
                                                                   accum_time)

        else:
            raise NotImplementedError

        consensus += cal_connectivity(model.components_, predict_h(model.components_)[0])

        out_list.append([cal_sparsity(model.components_), cal_sparsity(matrix_w),
                         cal_rss(matrix_w, model.components_, matrix_v),
                         cal_mse(matrix_w, model.components_, matrix_v),
                         cal_evar(cal_rss(matrix_w, model.components_, matrix_v), matrix_v)])

    out_list = np.array(out_list)

    out_list = list(np.mean(out_list, axis=0))

    consensus /= n_run

    out_list.append(cal_cophenetic(consensus))

    out_list.append(cal_dispersion(consensus))

    out_list.append(consensus)

    out_list.append(matrix_w_accum)

    out_list.append(matrix_h_accum)

    out_list.append(accum_time)

    summary = Summary(rank, *out_list)

    return summary


def clean_axis(axis):
    """

    Args:
        axis:

    Returns:

    """
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_consensus(matrix_consensus, prefix):
    """

    Args:
        matrix_consensus:
        prefix:

    Returns:

    """
    r_c = 1 - matrix_consensus

    fig = plt.figure(figsize=(13.9, 10))

    heatmap_gs = gridspec.GridSpec(1, 2, wspace=.01, hspace=0., width_ratios=[0.25, 1])

    y_link = sch.linkage(r_c, method='average')

    den_ax = fig.add_subplot(heatmap_gs[0, 0])

    den_d = sch.dendrogram(y_link, orientation='left', link_color_func=lambda k: 'black')

    clean_axis(den_ax)

    heatmap_ax = fig.add_subplot(heatmap_gs[0, 1])

    d_matrix = r_c[den_d['leaves'], :][:, den_d['leaves']]

    axi = heatmap_ax.imshow(d_matrix, interpolation='nearest', aspect='equal', origin='lower',
                            cmap='RdBu')

    clean_axis(heatmap_ax)

    color_bar = fig.colorbar(axi, fraction=0.046, pad=0.04, aspect=10)

    color_bar.set_label('Distance', fontsize=20)

    plt.savefig(f'{prefix}_consensus.svg', format='svg')

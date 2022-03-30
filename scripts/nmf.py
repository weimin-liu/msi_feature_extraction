import itertools
import math
import os
import pickle

import networkx as nx
import numpy as np
from matplotlib import gridspec
from mfe.feature import repeated_nmf
from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table
from mfe.util import CorSolver
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import NMF

from mfe.peak_picking import GLCMPeakRanking, de_flatten
import pandas as pd
import matplotlib.pyplot as plt
from settings import plot_params, path, tie_points

plt.rcParams.update(plot_params)

RANK = 10


def main(raw0,
         raw1,
         output,
         label):
    peak_th = 0.1
    xray = np.genfromtxt(os.path.join(path['xray'], 'X-Ray_pixel.txt'))[:, 0:3]
    A = pd.DataFrame(xray, columns=['px', 'py', 0])

    try:
        with open(os.path.join(path['results'], output), 'rb') as f:
            feature_table, error_table = pickle.load(f)
    except FileNotFoundError:
        spectra0 = msi_from_txt(raw0)
        spectra1 = msi_from_txt(raw1)
        for key in list(spectra0.keys()):
            spectra1[(-key[0], -key[1])] = spectra0[key]
        ref = get_ref_peaks(spectra1, peak_th=peak_th)[peak_th]
        feature_table, error_table = create_feature_table(spectra1, ref, normalization='median')
        with open(os.path.join(path['results'], output), 'wb') as f:
            pickle.dump([feature_table, error_table], f)

    feature_table0 = feature_table[feature_table['x'] < 0]

    feature_table0['x'] = -1 * feature_table0['x']
    feature_table0['y'] = -1 * feature_table0['y']
    cs0 = CorSolver()
    cs0.fit(tie_points[f'{label}_upper']['src'], tie_points[f'{label}_upper']['dst'])
    feature_table0[['px', 'py']] = cs0.transform(feature_table0[['x', 'y']])
    feature_table0 = feature_table0.merge(A, on=['px', 'py'])
    feature_table0 = feature_table0.drop(columns=['px', 'py'])

    feature_table1 = feature_table[feature_table['x'] > 0]
    cs1 = CorSolver()
    cs1.fit(tie_points[f'{label}_lower']['src'], tie_points[f'{label}_lower']['dst'])
    feature_table1[['px', 'py']] = cs1.transform(feature_table1[['x', 'y']])
    feature_table1 = feature_table1.merge(A, on=['px', 'py'])
    feature_table1 = feature_table1.drop(columns=['px', 'py'])

    feature_table1['x'] = feature_table1['x'] + feature_table0['x'].max()

    feature_table_com = pd.concat([feature_table0, feature_table1])

    feature_table1['x'] = feature_table1['x'] - feature_table0['x'].max()

    glcm0 = GLCMPeakRanking(q=8)
    glcm0.fit(feature_table_com, [1, 2, 3, 4, 5],
              [np.pi / 6, 0, -np.pi / 6, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4])

    mzs = glcm0.mzs_above_percentile(80)

    mzs = [float(mz) for mz in mzs]

    ims = list()

    for mz in mzs:
        im0 = de_flatten(feature_table0[['x', 'y']].to_numpy(), feature_table0[mz].to_numpy().flatten())
        im1 = de_flatten(feature_table1[['x', 'y']].to_numpy(), feature_table1[mz].to_numpy().flatten())
        if im0.shape[1] < im1.shape[1]:
            missing_width = im1.shape[1] - im0.shape[1]
            im0 = np.c_[im0, np.zeros([im0.shape[0], missing_width])]
        elif im0.shape[1] > im1.shape[1]:
            missing_width = im0.shape[1] - im1.shape[1]
            im1 = np.c_[im1, np.zeros([im1.shape[0], missing_width])]
        im1 = np.fliplr(im1)
        im = np.r_[im0, im1]
        ims.append(im)

    ims = [im.flatten() for im in ims]

    ims = np.array(ims)

    ims = ims.T

    img_shape = im.shape

    summary = repeated_nmf(ims, RANK, 30, init='random', max_iter=3000)

    return summary['W_accum'], pd.DataFrame(summary['H_accum'].T, index=mzs), img_shape, summary, feature_table_com


def spatial_fingerprints(W, H, min, max, img_shape, output):
    fig, axs = plt.subplots(W.shape[1], 2, figsize=(12, 12))

    for i in range(W.shape[1]):
        axs[i, 0].imshow(W[:, i].reshape(img_shape).T, aspect='auto')

        axs[i, 0].axis('off')

    for m in range(len(H.T)):
        for i in range(len(list(H.index))):
            axs[m, 1].plot([H.index[i], H.index[i]],
                           [0, H.iloc[i, m]], linewidth=0.3, color='black')
        axs[m, 1].set_xlim(min, max)
        axs[m, 1].spines['top'].set_visible(False)
        axs[m, 1].spines['right'].set_visible(False)
        axs[m, 1].spines['left'].set_visible(False)
        axs[m, 1].set_xticks([])
        # axs[m, 1].xaxis.set_tick_params(labelsize=10)
        axs[m, 1].yaxis.set_visible(False)
        axs[m, 1].set_ylim(0, 40)
        axs[m, 1].text(525, 10, f'NMF{m}', fontsize=12)
    axs[-1, 1].set_xticks(range(min, max, 20))
    axs[-1, 1].get_xaxis().tick_bottom()
    axs[-1, 1].set_xlabel('$\it{m/z}$')

    plt.tight_layout()

    plt.savefig(os.path.join(path['img'], output), format='svg')

    plt.show()


def create_network(summary, output, H, feature_table):
    output = 'a_graphml_0.9.graphml'
    C = summary['consensus']

    C = pd.DataFrame(C, index=H.index, columns=list(H.index))
    th = 0.9
    C[C <= th] = 0
    np.fill_diagonal(C.values, 0)
    G = nx.from_numpy_matrix(C.to_numpy())
    G = nx.relabel_nodes(G, lambda x: list(H.index)[x])
    for mz in list(G.nodes):
        avg_int = feature_table.loc[:, mz].mean()
        G.nodes[mz]['int'] = avg_int

    nx.write_graphml(G, os.path.join(path['results'], output))


if __name__ == "__main__":
    sterol_W, sterol_H, img_shape_s, summary_s, feature_table_s = main(os.path.join(path['raw'], 'SBB0-5cm_mz375-525.txt'),
                                                   os.path.join(path['raw'], 'SBB5-10cm_mz375-525.txt'),
                                                   'b_feature_error_pth_0.1.pkl',
                                                   'sterol')
    # alk_W, alk_H, img_shape_a, summary_a, feature_table_a = main(os.path.join(path['raw'], 'SBB0-5cm_mz520-580.txt'),
    #                                                os.path.join(path['raw'], 'SBB5-10cm_mz520-580.txt'),
    #                                                'a_feature_error_pth_0.1.pkl',
    #                                                'alkenone')
    # sterol_W0 =sterol_W[:,range(10)]
    # alk_W0 = alk_W[:, [3, 5, 6, 7, 9]]
    #
    # alk_W0 = [alk_W0[:,i].reshape(img_shape_a) for i in range(alk_W0.shape[1])]
    # alk_W0 = [m[:,1:-1] for m in alk_W0]
    # alk_W0 = [np.r_[m, np.zeros([1,45])] for m in alk_W0]
    # alk_W0 = [np.r_[ np.zeros([1,45]), m] for m in alk_W0]
    # alk_W0 = [m.flatten() for m in alk_W0]
    # alk_W0 = np.array(alk_W0).T
    #
    # corr = np.zeros([sterol_W0.shape[1], alk_W0.shape[1]])

    sterol = list(sterol_H.sort_values(by=2,ascending=False).index)[0:38]
    sterol = [x for x in sterol if int(x)%2!=0]

    sterol_f = feature_table_s[sterol]
    corr = sterol_f.corr()

    Y = sch.linkage(corr, method='average')
    fig, ax = plt.subplots(figsize=(3,6))
    denD = sch.dendrogram(Y, orientation='left', link_color_func=lambda k: 'black',leaf_label_func=lambda x:test[x])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()
    plt.show()

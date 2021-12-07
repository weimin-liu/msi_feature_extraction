import itertools

import networkx as nx
import numpy as np
import pandas as pd
import nimfa
from sklearn import preprocessing
from matplotlib import gridspec, pyplot as plt
import scipy.cluster.hierarchy as sch


class NMF:
    """
    Custom NMF calculator, taking in the feature table, detect the rank and then decompose based on the fittest rank.
    Also capable of calculate the consensus matrix and use it to build colocalization network.
    """

    def __init__(self):
        self._arr = np.array([])
        self._spot = np.array([])
        self._mzs = np.array([])
        self.nmf_fitted = None
        self.nmf_summary = None

    @property
    def mzs(self):
        return self._mzs

    @mzs.setter
    def mzs(self, x):
        self._mzs = x

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, x):
        self._arr = x

    @property
    def spot(self):
        return self._spot

    @spot.setter
    def spot(self, x):
        self._spot = x

    def rank_estimate(self, rank_cands):
        """
        first estimate the rank we want to use for nmf
        """
        nmf = nimfa.Nmf(self._arr, seed='random_vcol', max_iter=100, update='divergence', objective='div',
                        track_factor=True)
        summary = nmf.estimate_rank(rank_range=rank_cands, n_run=10, what='all')
        self.nmf_summary = summary
        coph = [summary[rank]['cophenetic'] for rank in rank_cands]
        evar = [summary[rank]['evar'] for rank in rank_cands]
        plt.plot(rank_cands, coph, 'o-', label='Cophenetic correlation', linewidth=2)
        plt.plot(rank_cands, evar, 'o-', label='Explained variance', linewidth=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, numpoints=1)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def clean_axis(ax):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def compute_nmf(self, *args, **kwargs):
        nmf = nimfa.Nmf(self._arr, *args, **kwargs)
        self.nmf_fitted = nmf()

    def consensus(self, figsize=(13.9, 10)):
        fig = plt.figure(figsize=figsize)
        heatmapGS = gridspec.GridSpec(1, 2, wspace=.01, hspace=0., width_ratios=[0.25, 1])

        C = 1 - self.nmf_fitted.fit.consensus()
        Y = sch.linkage(C, method='average')

        denAX = fig.add_subplot(heatmapGS[0, 0])
        denD = sch.dendrogram(Y, orientation='left', link_color_func=lambda k: 'black')
        self.clean_axis(denAX)

        heatmapAX = fig.add_subplot(heatmapGS[0, 1])
        D = C[denD['leaves'], :][:, denD['leaves']]
        axi = heatmapAX.imshow(D, interpolation='nearest', aspect='equal', origin='lower', cmap='RdBu')
        self.clean_axis(heatmapAX)

        cb = fig.colorbar(axi, fraction=0.046, pad=0.04, aspect=10)
        cb.set_label('Distance', fontsize=20)
        plt.show()

    def to_graphml(self, path, th=0.9):
        """
        save the molecular networking to .graphml format
        path: destination file
        th: threshold for the consensus score when building the network. A lower value would allow the building of larger sub-networks
        """
        consensus = pd.DataFrame(self.nmf_fitted.fit.consensus(), index=self._mzs, columns=self._mzs)
        consensus.values[[np.arange(len(consensus))] * 2] = np.nan
        consensus = consensus.stack().reset_index()
        consensus = consensus[consensus[0] >= th]
        G = nx.from_pandas_edgelist(consensus, source='level_0', target='level_1')
        nx.write_graphml(G, path)

    def from_feature_table(self, path: str, th=0.4):
        """
        this is not an alternative builder. It's only callable after the instance has been initialized.
        path: path to the feature table
        th: threshold for the sparsity of the ion image on the slide. A lower value will allow less ions to be include in the following process
        """
        df = pd.read_csv(path)
        df = df.astype(float)
        self._spot = df[['x', 'y']].to_numpy()
        df = df.drop(columns=['x', 'y'])
        df = df.replace(np.nan, 0)

        sparsity = (df == 0).astype(int).sum(axis=0) / len(df)
        sparsity = sparsity[sparsity <= th]
        self._mzs = np.array(sparsity.index.to_list())
        df = df[sparsity.index.to_list()]
        arr = df.to_numpy()
        arr = arr - np.min(arr)
        self._arr = preprocessing.Normalizer().fit_transform(arr)

    def from_arr(self, mzs: np.ndarray, images: np.ndarray):
        """
        """
        self._mzs = mzs

        self._arr = images


def rank_estimate():
    rank_cands = list(range(2, 20))
    rank_cands.extend(list(range(20, 60, 10)))
    nmf = nimfa.Snmf(im, seed='random_vcol', max_iter=1000, update='divergence', objective='div',version='r',
                    track_factor=True)
    summary = nmf.estimate_rank(rank_range=rank_cands, n_run=10, what='all')
    coph = [summary[rank]['cophenetic'] for rank in rank_cands]
    evar = [summary[rank]['evar'] for rank in rank_cands]
    plt.plot(rank_cands, coph, 'o-', label='Cophenetic correlation', linewidth=2)
    plt.plot(rank_cands, evar, 'o-', label='Explained variance', linewidth=2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, numpoints=1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np_load = np.load(r"../../../examples/SBB5-10cm_mz520-580_structured_data.npz")
    mzs = np_load['mzs']
    im = np_load['im']
    im = im.T
    im[np.isnan(im)] = 0

    # snmf = nimfa.Snmf(im, seed="random_c", rank=7, max_iter=1000, version='r', eta=1.,
    #               beta=1e-4, i_conv=10, w_min_change=0)
    # # nmf = nimfa.Nmf(im, seed='random_vcol', max_iter=1000, update='divergence', objective='div',
    # #                 track_factor=True, n_run=30, rank=7)
    # nmf_fitted = snmf()
    #
    # # consensus = pd.DataFrame(nmf_fitted.fit.consensus(), index=mzs, columns=mzs)
    # # consensus.values[[np.arange(len(consensus))] * 2] = np.nan
    # # consensus = consensus.stack().reset_index()
    # # consensus = consensus[consensus[0] >= 0.5]
    # # G = nx.from_pandas_edgelist(consensus, source='level_0', target='level_1')
    # # nx.write_graphml(G, '../../../examples/SBB0-5cm_mz520-580_network.graphml')
    # components = pd.DataFrame(nmf_fitted.fit.coef().T, index=list(mzs))
    # #
    # basis = nmf_fitted.fit.basis()
    # fig, ax = plt.subplots(1, 7)
    # for m in range(len(basis.T)):
    #     image = basis[:, m].reshape(209, 37)
    #     ax[m].imshow(image, cmap='jet')
    #     ax[m].axis('off')
    #     ax[m].set_title(f'{m}')
    # plt.tight_layout()
    # plt.show()
    #
    # group0 = components.iloc[:, 1]
    # group0 = group0[group0 >= 0.3]
    # group0 = list(group0.index)
    # group0_images = list()
    # for mz in group0:
    #     mz_index = np.where(mzs == mz)
    #     group0_images.append(im[:, mz_index].flatten())
    # group0_images = np.array(group0_images)
    # import numpy as np
    #
    # import itertools
    #
    # group0_mz_comb = list(itertools.combinations(group0, 2))
    # group0_mz_comb = pd.DataFrame(group0_mz_comb)
    # group0_mz_comb['d'] = group0_mz_comb.iloc[:, 0] - group0_mz_comb.iloc[:, 1]
    # group0_mz_comb['d'] = group0_mz_comb['d'].abs()
    #
    # txt_path = r"../../../examples/SBB0-5cm_mz520-580.txt"
    # acc_mz, _ = get_accmz(txt_path, group0)
    # acc_mz = list(acc_mz.values())
    # k_mz = pd.DataFrame(acc_mz, columns=['n_mz'])
    # k_mz['h2_k'] = k_mz * 2 / 2.01565
    # k_mz['h2_d'] = k_mz['n_mz'].astype(int) - k_mz['h2_k']
    # k_mz = k_mz[k_mz.n_mz.astype(int) % 2 != 0]
    # k_mz = k_mz.reset_index(drop=True)
    #
    # plt.scatter(k_mz['n_mz'], k_mz['h2_d'])
    # plt.xlabel('m/z (Da)')
    # plt.ylabel('KMD (CH2)')
    # plt.show()
    #
    # group0_mz_combo = pd.DataFrame(itertools.combinations(list(k_mz['n_mz']), 2))
    # group0_mz_combo_dict = k_mz[['n_mz', 'h2_d']].set_index('n_mz')['h2_d'].to_dict()
    # for i in range(len(group0_mz_combo)):
    #     mz_diff = group0_mz_combo.iloc[i, 0] - group0_mz_combo.iloc[i, 1]
    #     k_dff = group0_mz_combo_dict[group0_mz_combo.iloc[i, 0]] - group0_mz_combo_dict[group0_mz_combo.iloc[i, 1]]
    #     group0_mz_combo.loc[i, 'distance'] = pow((mz_diff * mz_diff + k_dff * k_dff), 1 / 2)
    #
    # group0_mz_combo = group0_mz_combo.sort_values(by='distance')
    # group0_mz_combo = group0_mz_combo[group0_mz_combo['distance'] < 4.5]
    # mzs1 = np.unique(group0_mz_combo[[0, 1]].to_numpy().flatten())
    # group0_mz_combo = group0_mz_combo.drop(columns=['distance'])
    # group0_mz_combo = group0_mz_combo.round(2)
    # group0_mz_combo = group0_mz_combo.rename(columns={0: 'level_0', 1: 'level_1'})
    # group0_mz_combo.to_csv("../../../examples/consensus.csv")
    #
    # for i in range(len(k_mz)):
    #     if k_mz.loc[i, 'n_mz'] not in mzs1:
    #         k_mz.loc[i, 'n_mz'] = np.nan
    # k_mz = k_mz.dropna()
    # k_mz = k_mz.reset_index(drop=True)
    #
    # plt.scatter(k_mz['n_mz'], k_mz['h2_d'])
    # plt.xlabel('m/z (Da)')
    # plt.ylabel('KMD (CH2)')
    # plt.show()

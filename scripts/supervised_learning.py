import math
import os

from matplotlib.lines import Line2D
from mfe.peak_picking import GLCMPeakRanking
from scipy.stats import zscore
import tqdm
from scipy.stats import spearmanr
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, train_test_split, KFold, \
    TimeSeriesSplit
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor

import random
from matplotlib import pyplot as plt
from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table
from mfe.time import from_year_fraction
from mfe.util import CorSolver
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import networkx as nx
import pickle
from settings import plot_params, path, tie_points
import seaborn as sb
from scipy import signal
from dtaidistance import dtw

from util import get_all_instrumental_data

plt.rcParams.update(plot_params)
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)



def main(table, label):
    xray = np.genfromtxt(os.path.join(path['xray'], 'X-Ray_pixel.txt'))[:, 0:3]
    xray_warped = np.genfromtxt(os.path.join(path['xray'], 'X-Ray_pixel_warped.txt'))[:, 0:2]
    depth = np.genfromtxt(os.path.join(path['xray'], 'X-Ray_depth.txt'))[:, 1]
    A = pd.DataFrame(np.hstack([xray, xray_warped, depth.reshape(-1, 1)]),
                     columns=['px', 'py', 'gs', 'wpx', 'wpy', 'depth'])
    age_model = np.genfromtxt('./data/age_model.csv')

    with open(os.path.join(path['results'], table), 'rb') as f:
        feature_table, error_table = pickle.load(f)

    feature_table0 = feature_table[feature_table['x'] < 0]

    feature_table0['x'] = -1 * feature_table0['x']
    feature_table0['y'] = -1 * feature_table0['y']
    cs0 = CorSolver()
    cs0.fit(tie_points[f'{label}_upper']['src'], tie_points[f'{label}_upper']['dst'])
    feature_table0[['px', 'py']] = cs0.transform(feature_table0[['x', 'y']])
    feature_table0 = feature_table0.merge(A, on=['px', 'py'])
    feature_table0['age'] = np.interp(feature_table0['depth'], age_model[:, 0], age_model[:, 1])

    feature_table1 = feature_table[feature_table['x'] > 0]
    cs1 = CorSolver()
    cs1.fit(tie_points[f'{label}_lower']['src'], tie_points[f'{label}_lower']['dst'])
    feature_table1[['px', 'py']] = cs1.transform(feature_table1[['x', 'y']])
    feature_table1 = feature_table1.merge(A, on=['px', 'py'])
    feature_table1['age'] = np.interp(feature_table1['depth'], age_model[:, 0], age_model[:, 1])

    feature_table1['x'] = feature_table1['x'] + feature_table0['x'].max()
    feature_table_com = pd.concat([feature_table0, feature_table1])
    feature_table1['x'] = feature_table1['x'] - feature_table0['x'].max()
    gs = feature_table_com['gs']
    feature_table_com = feature_table_com.iloc[:, :-7]
    feature_table_com[0] = gs
    glcm0 = GLCMPeakRanking(q=8)
    glcm0.fit(feature_table_com, [1, 2, 3, 4, 5],
              [np.pi / 6, 0, -np.pi / 6, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4])

    mzs = glcm0.mzs_above_percentile(80)

    mzs = [float(mz) for mz in mzs]
    mzs.append('age')

    feature_table0 = feature_table0[mzs]
    feature_table1 = feature_table1[mzs]

    f = pd.concat([feature_table0, feature_table1])
    age_tie = pd.read_csv('./data/aget_tie.csv', delimiter=';').to_numpy()
    f['age'] = np.interp(f['age'].to_numpy(), age_tie[:, 0], age_tie[:, 1])
    f['age'] = f['age'].map(from_year_fraction)

    f = f.set_index('age')
    f_q = f.resample('Q').agg(['mean', 'count'])

    for i in range(1, len(f_q.T), 2):
        f_q.iloc[:, i] = f_q.iloc[:, i].map(lambda x: np.nan if x <= 10 else x)

    f_q = f_q.dropna()

    f_q = f_q.iloc[:, [i for i in range(0, len(f_q.T), 2)]]

    return f, f_q


if __name__ == "__main__":
    f, f_q = main('a_feature_error_pth_0.1.pkl', 'alkenone')

    #
    instrumental_data = get_all_instrumental_data(r'./data/environment')
    # #
    alkenone = [551.5166,552.5198,553.5317,554.5349,565.5317,566.5349,567.5467,568.5499,569.5263]
    # #
    # chlorophyll = [557.2521, 561.2467,573.2467,575.2263,535.2704,529.2199,543.2371,573.2166, 559.2317, 533.2521, 559.2586]
    # sterol = list(pd.read_clipboard(decimal=',').to_numpy().flatten())

    f_q = f_q[alkenone]

    # G = nx.read_graphml('./data/sterodial.graphml')
    #
    # sterol = [float(G.nodes[node]['name']) for node in list(G.nodes)]
    #
    # f = f[chlorophyll]
    #
    # enso = pd.read_csv(os.path.join(path['env'], 'meiv2.data.txt'), delimiter='\t')
    # enso = enso.set_index('Year')
    # age = list(enso.index)
    #
    # enso = enso.to_numpy().flatten()
    # enso = pd.DataFrame(enso, columns=['value'])
    # m = 0
    # for i in range(1, len(enso)):
    #     mt = i % 12
    #     if mt == 0:
    #         m += 1
    #         mt = 12
    #     enso.loc[i, 'age'] = datetime.datetime(age[m], mt, 1)
    # enso['age'] = enso['age'] + pd.DateOffset(months=1)
    # enso = enso.set_index('age')
    # enso = enso[1:]
    # enso = enso.apply(lambda x: x.str.replace(',', '.'))
    # enso = enso.astype(float)
    #
    # enso = enso['value'].resample('M').mean()
    #
    # f['enso'] = enso
    # # f_yr = f
    # f_yr = f.rolling(window=12).mean()
    # f_yr = f_yr.dropna()
    # # dst = []
    # # for i in range(len(f.iloc[:, 0:-1].T)):
    # #     dst.append(dtw.distance(zscore(f.iloc[:, i]), zscore(f.iloc[:, -1]), window=1))
    # # dst = pd.DataFrame(dst, index=f.iloc[:, 0:-1].columns)

    f_q['enso'] = instrumental_data['temperature'].iloc[:,2].resample('Q').mean()
    f_q = f_q.dropna()

    X = f_q.dropna().iloc[:, 0:-1]

    y = f_q.dropna().iloc[:, -1]
    #
    tscv = TimeSeriesSplit()
    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("net", ElasticNet(max_iter=1e7))])
    param_grid = {
        'net__l1_ratio': np.arange(0, 1, 0.01)[1:],
        'net__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    }

    stanol_stenol = f.loc[:, 553.5317] / (f.loc[:, 553.5317] + f.loc[:, 551.5166])
    stanol_stenol = stanol_stenol.replace(0, np.nan)
    stanol_stenol = stanol_stenol.replace(np.inf, np.nan)
    stanol_stenol = stanol_stenol.dropna()
    stanol_stenol = stanol_stenol.resample('Q').agg(['mean', 'count'])
    stanol_stenol = stanol_stenol[stanol_stenol['count'] > 20]
    stanol_stenol = stanol_stenol['mean']

    sh = GridSearchCV(pipe, param_grid, cv=tscv, scoring='neg_mean_squared_error').fit(X, y)
    test = pd.DataFrame(sh.predict(f_q.dropna().iloc[:, 0:-1]), index=f_q.index)
    fig, ax = plt.subplots()
    ax.plot(zscore(test), color='blue', label='Modeled from A-NMF6',alpha=0.7,marker='o',markersize=5)
    ax.plot(zscore(f_q['enso']) + 3 , color='red', label='SST (°C, 0-30m)',alpha=0.7,marker='o',markersize=5)
    ax.plot(zscore(stanol_stenol) + 6, color='black', label="$U^{k'}_{37}$",alpha=0.7,marker='o',markersize=5)
    ax.set_xlim(datetime.datetime(1984, 1, 1), datetime.datetime(2008, 12, 30))
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.legend(loc='best',bbox_to_anchor=(0.5,0.9))
    plt.xlabel('Year AD')
    plt.tight_layout()
    plt.show()
    coef = pd.DataFrame(sh.best_estimator_.steps[1][1].coef_, index=sterol)



    # from sklearn.preprocessing import StandardScaler
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=10)
    # pc = pca.fit_transform(StandardScaler().fit_transform(f.iloc[:,:-1]))
    # pc = pd.DataFrame(pc,index=f.index)
    # test = pc.iloc[:,1]
    # fig, [ax0, ax1] = plt.subplots(2, sharex=True)
    # ax0.plot(zscore(test), color='blue', alpha=0.6, linewidth=1, label='monthly')
    # ax0.plot(zscore(test.rolling(window=3).mean().dropna()), color='red', alpha=0.6, linewidth=1, label='3M moving avg')
    # ax0.legend(loc='upper left', fontsize=8)
    # # ax0.set_ylim([5,-5])
    # ax0.set_ylabel('Zscore')
    #
    # for i in range(len(list(f.index))):
    #     ax1.plot([f.index[i], f.index[i]],
    #              [0, f.iloc[i, -1]], linewidth=0.3, color='black')
    #
    # plt.xlim(datetime.datetime(1984, 1, 1), datetime.datetime(2008, 12, 30))
    # plt.xlabel('Year')
    # ax1.set_ylabel('MEI.v2')
    # plt.savefig('./data/img/enso.pdf', format='pdf')
    # plt.suptitle('Median-normalized intensity of 548.2435 $\it{vs}$. MEI.v2')
    #
    # plt.show()

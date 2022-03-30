import itertools
import pickle
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mfe.feature import repeated_nmf
from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table
from mfe.peak_picking import de_flatten, GLCMPeakRanking
from mfe.time import from_year_fraction
from mfe.util import CorSolver
from scipy.stats import zscore

from settings import path, tie_points
from util import get_all_instrumental_data
from dtaidistance import dtw
from settings import plot_params

plt.rcParams.update(plot_params)


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


if __name__ == "__main__":
    RANK = 10
    peak_th = 0.1
    label = 'alkenone'
    raw0 = 'SBB0-5cm_mz520-580.txt'
    raw1 = 'SBB5-10cm_mz520-580.txt'
    output = 'a_feature_error_pth_0.1.pkl'

    xray = np.genfromtxt(os.path.join(path['xray'], 'X-Ray_pixel.txt'))[:, 0:3]
    xray_warped = np.genfromtxt(os.path.join(path['xray'], 'X-Ray_pixel_warped.txt'))[:, 0:2]
    depth = np.genfromtxt(os.path.join(path['xray'], 'X-Ray_depth.txt'))[:, 1]
    A = pd.DataFrame(np.hstack([xray, xray_warped, depth.reshape(-1, 1)]),
                     columns=['px', 'py', 'gs', 'wpx', 'wpy', 'depth'])
    age_model = np.genfromtxt('./data/age_model.csv')

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

    feature_table_com = feature_table_com.iloc[:, :-3].rename(columns={'gs': 0})

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

    spatial_fingerprints(summary['W_accum'], pd.DataFrame(summary['H_accum'].T, index=mzs), 520, 580, img_shape,
                         'a_nmf.svg')

    sel = [1,4,5,8,9]
    W0 = summary['W_accum'][:, sel]
    H0 = pd.DataFrame(summary['H_accum'].T[:, sel], index=mzs)

    feature_table0['age'] = np.interp(feature_table0['depth'], age_model[:, 0], age_model[:, 1])
    feature_table1['age'] = np.interp(feature_table1['depth'], age_model[:, 0], age_model[:, 1])

    age0 = feature_table0['age']
    age1 = feature_table1['age']
    feature_table0 = feature_table0[mzs]
    feature_table1 = feature_table1[mzs]
    for h in range(len(H0.T)):
        feature_table0[f'{h}'] = (H0.iloc[:, h] * feature_table0).sum(axis=1)
        feature_table1[f'{h}'] = (H0.iloc[:, h] * feature_table1).sum(axis=1)
    nmf0 = feature_table0.iloc[:, -5:]
    nmf0['yr'] = age0
    nmf1 = feature_table1.iloc[:, -5:]
    nmf1['yr'] = age1

    nmf_c = pd.concat([nmf0, nmf1])
    nmf_c['yr'] = nmf_c['yr'].map(from_year_fraction)

    nmf_c = nmf_c.set_index('yr')
    nmf_c = nmf_c.resample('M').agg(['mean', 'count'])
    for i in range(1, len(nmf_c.T), 2):
        nmf_c.iloc[:, i] = nmf_c.iloc[:, i].map(lambda x: np.nan if x <= 10 else x)

    nmf_c = nmf_c.dropna()

    nmf_c = nmf_c.iloc[:, [i for i in range(0, len(nmf_c.T), 2)]]

    for m in range(len(nmf_c.T)):
        nmf_c = nmf_c.rename(columns={f'{m}': f'A-NMF{sel[m]}'})

    nmf_c.to_csv('./data/a_nmf.csv')
    nmf_b = pd.read_csv('./data/b_nmf.csv')[2:].set_index('Unnamed: 0')
    nmf_b.index = pd.to_datetime(nmf_b.index)
    nmf = pd.concat([nmf_b, nmf_c], axis=1)
    nmf = nmf.dropna()
    #
    instrumental_data = get_all_instrumental_data(r'./data/environment')
    instrumental_data['oxygen'] = instrumental_data['oxygen'].iloc[:, :-1]
    del instrumental_data['chlora']
    # del instrumental_data['temperature']
    # inst_sel = [2, 5, 8, 10, 12]
    for key in instrumental_data:
        instrumental_data[key] = instrumental_data[key].iloc[:,:-4]
        instrumental_data[key] = instrumental_data[key].resample('M').mean()
        instrumental_data[key] = instrumental_data[key].interpolate(method='linear')
    env = pd.concat(instrumental_data.values(), axis=1)
    nmf = nmf[(nmf.index >= min(env.index)) & (nmf.index <= max(env.index))]
    env = env[(env.index >= min(nmf.index)) & (env.index <= max(nmf.index))]

    nmf = nmf.astype(float)
    dst = np.zeros([env.shape[1], nmf.shape[1]])
    for m, n in itertools.product(range(env.shape[1]), range(nmf.shape[1])):
        dst[m, n] = dtw.distance_fast(zscore(-nmf.iloc[:, n].rolling(window=3).mean().dropna().to_numpy()),
                                      zscore(env.iloc[:, m].rolling(window=3).mean().dropna().to_numpy()), window=6)

    dst = pd.DataFrame(dst)
    for i in range(14):
        dst = dst.rename(columns={i: nmf.columns[i]})

    fig, ax = plt.subplots(figsize=(8, 6))
    axi = ax.imshow(dst, cmap='RdBu', aspect='auto', vmax=16)
    ax.set_xticks(ticks=range(14), labels=nmf.columns, rotation=90)
    ax.set_yticks(ticks=[4.5, 14.5, 24.5, 34.5, 44.5, 54.5,64.5],
                  labels=['Salinity', 'Nitrite', 'Silicate', 'Oxygen', 'Temperature','Phosphate', 'Nitrate'])
    cb = fig.colorbar(axi, fraction=0.046, pad=0.04, aspect=10)
    cb.set_label('DTW Distance (Reversed)')
    cb.set_ticks([12, 16])
    ax.set_xlabel('Spatial fingerprints')
    ax.set_ylabel('Environmental parameters')
    plt.tight_layout()
    plt.show()

    plt.plot(zscore(nmf['A-NMF6'].rolling(window=3).mean().dropna()), label='A-NMF6')
    plt.plot(zscore(env.iloc[:, 80].rolling(window=3).mean().dropna()), label='Temperature (50-200 m)')
    plt.xlabel('Year')
    plt.ylabel('Zscore')
    plt.legend()
    plt.tight_layout()
    plt.show()

import concurrent
import pickle

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from mfe.calibration import quadric_calibration
from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table

from mfe.peak_picking import GLCMPeakRanking, de_flatten
from mfe.util import CorSolver
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics.pairwise import cosine_similarity
from settings import tie_points

from settings import plot_params

plt.rcParams.update(plot_params)



if __name__ == "__main__":

    ft = './data/b_feature_table_0.1.pkl'
    try:
        with open(ft, 'rb') as f:
            feature_table = pickle.load(f)
    except FileNotFoundError:
        spectra0 = msi_from_txt("./data/raw/SBB0-5cm_mz520-580.txt")
        spectra1 = msi_from_txt("./data/raw/SBB5-10cm_mz520-580.txt")

        for key in list(spectra0.keys()):
            spectra1[(-key[0], -key[1])] = spectra0[key]
        ref = get_ref_peaks(spectra1, peak_th=0.1)
        ref = ref[0.1]

        feature_table, error_table = create_feature_table(spectra1, ref, normalization='median')

        with open(ft, 'wb') as f:
            pickle.dump(feature_table, f)

    feature_table0 = feature_table[feature_table['x'] < 0]

    feature_table0['x'] = -1 * feature_table0['x']
    feature_table0['y'] = -1 * feature_table0['y']


    cs = CorSolver()
    cs.fit(tie_points['sterol_upper']['src'], tie_points['sterol_upper']['dst'])
    feature_table0[['px', 'py']] = cs.transform(feature_table0[['x', 'y']])
    xray = np.genfromtxt('./data/xray/X-Ray_pixel.txt')[:, 0:3]
    A = pd.DataFrame(xray,
                     columns=['px', 'py', 0])
    feature_table0 = feature_table0.merge(A, on=['px', 'py'])
    feature_table0 = feature_table0.drop(columns=['px', 'py'])

    feature_table1 = feature_table[feature_table['x'] > 0]

    cs = CorSolver()
    cs.fit(tie_points['sterol_lower']['src'], tie_points['sterol_lower']['dst'])
    feature_table1[['px', 'py']] = cs.transform(feature_table1[['x', 'y']])
    feature_table1 = feature_table1.merge(A, on=['px', 'py'])
    feature_table1 = feature_table1.drop(columns=['px', 'py'])

    feature_table1['x'] =  feature_table1['x'] + feature_table0['x'].max()

    feature_table_com = pd.concat([feature_table0,feature_table1])

    glcm0 = GLCMPeakRanking(q=8)
    glcm0.fit(feature_table_com, [1, 2, 3, 4, 5], [np.pi / 6, 0, -np.pi / 6, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4])

    results = glcm0.results
    results = pd.DataFrame(results)
    results = results.dropna()
    results.index = np.array(list(feature_table0.columns))[list(results.index)]

    similarities = pd.DataFrame(cosine_similarity(results),index=results.index, columns=list(results.index))

    glcm0.fancy_overview('./data/img/b_glcm.svg',[-0.2,0.2],[-0.2,0.2])
    mz = glcm0.mzs_above_percentile(80)

    percentile = [99,95,90,85,80,75,70]
    mzs = glcm0.mz_at_percentile(percentile)
    fig, axs = plt.subplots(len(percentile))
    m=0
    for i in range(len(percentile)):
        mz =mzs[i]
        im0 = de_flatten(feature_table0[['x', 'y']].to_numpy(), feature_table0[float(mz)].to_numpy().flatten(),
                         interpolation='None')
        im1 = de_flatten(feature_table1[['x', 'y']].to_numpy(), feature_table1[float(mz)].to_numpy().flatten(),
                         interpolation='None')
        im1 = np.fliplr(im1)
        if im0.shape[1] < im1.shape[1]:
            missing_width = im1.shape[1] - im0.shape[1]
            im0 = np.c_[im0, np.zeros([im0.shape[0], missing_width])]
        elif im0.shape[1] > im1.shape[1]:
            missing_width = im0.shape[1] - im1.shape[1]
            im1 = np.c_[im1, np.zeros([im1.shape[0], missing_width])]
        im = np.r_[im0, im1]
        axs[m].imshow(im.T, interpolation='None')
        axs[m].set_title(f'{percentile[i]}th', x=0.08, y=1.0, pad=-10, fontsize=10, color='w')
        axs[m].axis('off')
        m += 1
    plt.tight_layout()
    plt.savefig(f'./data/img/b_percentile_after_glcm_{percentile[0]}.svg', format='svg')
    plt.show()
    metric = similarities['x']
    top5 = list(metric.sort_values(ascending=False).index)[1:6]
    bottom5 = list(metric.sort_values( ascending=True).index)[0:5]
    for ls in [top5, bottom5]:
        fig, axs = plt.subplots(len(ls))
        m = 0
        for i in ls:
            i = round(float(i),4)
            im0 = de_flatten(feature_table0[['x','y']].to_numpy(), feature_table0[i].to_numpy().flatten(),
                             interpolation='None')
            im1= de_flatten(feature_table1[['x','y']].to_numpy(), feature_table1[i].to_numpy().flatten(),
                            interpolation='None')
            im1 = np.fliplr(im1)
            if im0.shape[1] < im1.shape[1]:
                missing_width = im1.shape[1] - im0.shape[1]
                im0 = np.c_[im0, np.zeros([im0.shape[0], missing_width])]
            elif im0.shape[1] > im1.shape[1]:
                missing_width = im0.shape[1] - im1.shape[1]
                im1 = np.c_[im1, np.zeros([im1.shape[0], missing_width])]
            im = np.r_[im0, im1]
            axs[m].imshow(im.T, interpolation='None')
            axs[m].set_title(f'{round(i, 3)}',x=0.08, y=1.0,pad=-10,fontsize=10,color='w')
            axs[m].axis('off')
            m += 1
        plt.tight_layout()
        plt.savefig(f'./data/img/a_example_after_glcm_{ls[0]}.svg', format='svg')
        plt.show()

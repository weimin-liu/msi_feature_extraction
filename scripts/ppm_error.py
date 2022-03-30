"""
Fig.3 showing the map of recovered TIC percentage and the mean absolute mass drift across the mass range
"""
import concurrent

import numpy as np
from mfe.calibration import quadric_calibration
from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table, peak_alignment_evaluation
import matplotlib.pyplot as plt
from tqdm import tqdm

from settings import plot_params
from mfe.peak_picking import de_flatten

plt.rcParams.update(plot_params)


def tmp(txt0, txt1, pth):
    spectra0 = msi_from_txt(txt0)
    spectra1 = msi_from_txt(txt1)

    for key in list(spectra1.keys()):
        spectra0[(-key[0], -key[1])] = spectra1[key]

    ref = get_ref_peaks(spectra0, peak_th=pth)

    ref = ref[pth]

    feature_table, error_table = create_feature_table(spectra0, ref)

    coverage = peak_alignment_evaluation(spectra0, feature_table)
    return {
        'feature': feature_table,
        'error': error_table,
        'coverage': coverage
    }


if __name__ == "__main__":
    # mzs_alkenone = [557.2523, 533.2523, 553.5319, 537.2261, 535.2104, 522.5972, 550.6285, 561.2472, 559.2316, 573.2472,
    #        569.4540, 569.2523, 551.2417]
    # mzs_sterol = [413.2662, 441.2975, 409.2713, 429.2400, 477.4642, 435.3597, 433.3805, 391.3547, 407.3284, 522.5972, 393.2975]

    sterol = tmp("./data/raw/SBB0-5cm_mz375-525.txt", "./data/raw/SBB5-10cm_mz375-525.txt", 0.1) #0.79 %tic
    alkenone = tmp("./data/raw/SBB0-5cm_mz520-580.txt", "./data/raw/SBB5-10cm_mz520-580.txt", 0.1)  # 0.78 %tic

    tic_per = list()
    for result in [ alkenone, sterol]:
        cov_table = result['coverage']
        cov_table0 = cov_table[cov_table['x'] > 0]
        cov_table1 = cov_table[cov_table['x'] < 0]
        im0 = de_flatten(cov_table0[['x', 'y']].to_numpy(),
                         cov_table0.drop(columns=['x', 'y']).to_numpy().flatten(),
                         stretch_contrast=False,
                         interpolation='None'
                         )
        im1 = de_flatten(cov_table1[['x', 'y']].abs().to_numpy(),
                         cov_table1.drop(columns=['x', 'y']).to_numpy().flatten(),
                         stretch_contrast=False,
                         interpolation='None'
                         )
        if im0.shape[1] < im1.shape[1]:
            missing_width = im1.shape[1] - im0.shape[1]
            im0 = np.c_[im0, np.zeros([im0.shape[0], missing_width])]
        elif im0.shape[1] > im1.shape[1]:
            missing_width = im0.shape[1] - im1.shape[1]
            im1 = np.c_[im1, np.zeros([im1.shape[0], missing_width])]
        im1 = np.fliplr(im1)
        im = np.r_[im0, im1]
        im = im.T
        tic_per.append(im)

    fig, ax = plt.subplots(2, 1,figsize=(8,1.5))

    ax[0].imshow(tic_per[0], interpolation='None', vmin=0.3, vmax=0.8,aspect='auto')
    ax[0].axis('off')

    im = ax[1].imshow(tic_per[1], interpolation='None', vmin=0.3,
                      vmax=0.8,aspect='auto')
    ax[1].axis('off')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig('./data/img/tic_coverage.svg', format='svg')
    # plt.tight_layout()
    plt.show()

    alkenone['mzs'] = list(alkenone['error'].drop(columns=['x', 'y']).columns)

    sterol['mzs'] = list(sterol['error'].drop(columns=['x', 'y']).columns)

    test = alkenone['error'].drop(columns=['x', 'y']).to_numpy()

    test = np.abs(test)

    plt.scatter(alkenone['mzs'], np.nanmean(test, axis=0), s=20, alpha=0.5)
    plt.xlabel('$\it{m/z}$')
    plt.yticks([1, 2, 3, 4, 5])
    plt.ylabel('Mean absolute mass drift (ppm)')
    plt.tight_layout()
    plt.savefig('./data/img/alkenone_mass_drift.svg', format='svg')
    plt.show()

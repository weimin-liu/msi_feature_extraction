"""
Fig.1 in the manuscript, drawing the result of kernel density estimation,
"""

import numpy as np
from KDEpy import FFTKDE
from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table
import matplotlib.pyplot as plt
import seaborn as sb
from settings import plot_params
plt.rcParams.update(plot_params)

if __name__ == "__main__":
    spectra0 = msi_from_txt("./data/raw/SBB0-5cm_mz375-525.txt")

    mzs = [[553.53, 553.54], [551.51, 551.52], [567.54, 567.55], [565.53, 565.54]]

    mzs_all = [spec._peaks_mz for spec in spectra0.values()]

    mzs_all = np.concatenate(mzs_all).ravel()

    mzs_all = np.sort(mzs_all)

    mz_0 = mzs_all[(mzs_all >=567.54 ) & (mzs_all <=567.55)]

    sb.histplot(mzs_all)

    plt.xlabel('$\it{m/z}$')

    plt.tight_layout()

    plt.savefig('./img/peak_count_in_mzall.svg', format='svg')

    plt.show()

    mzs = mzs_all[(mzs_all >= 551) & (mzs_all < 552)]

    x, y = FFTKDE(kernel='gaussian', bw='ISJ').fit(mzs).evaluate()

    y = y / np.max(y)

    from scipy import signal

    peaks, _ = signal.find_peaks(y, prominence=(0.1, None))
    plt.plot(x_, y_)
    plt.plot(x_[peaks], y_[peaks], "x")
    plt.xlabel('$\it{m/z}$')

    plt.tight_layout()

    plt.savefig('./data/img/reference_peaks.svg', format="svg")
    plt.show()


from mfe.src.from_txt import msi_from_txt
import numpy as np
from tqdm import tqdm
import seaborn as sb
import matplotlib.pyplot as plt
from KDEpy import FFTKDE

if __name__ == "__main__":
    raw_txt_path = r'../../examples/SBB5-10cm_mz520-580.txt'
    spectra = msi_from_txt(raw_txt_path)

    # get all mzs from the sample
    mzs = [spec._peaks_mz for spec in spectra.values()]
    mzs = np.concatenate(mzs).ravel()
    mzs = np.sort(mzs)
    n = len(mzs)
    d = np.diff(mzs)
    RESOLUTION = 1e5
    d_th = 0.2 * 500 / RESOLUTION
    isnew = d > d_th

    cluster = list()
    i = 0
    while i < len(isnew) - 1:
        print(i)
        cand = list()
        while not isnew[i]:
            cand.append(i)
            i += 1
        if len(cand)!=0:
            cluster.append(cand)
        i+=1
    cluster = [c for c in cluster if len(c)>=1000]

    cnzs = [mzs[idx] for idx in cluster[55]]

    x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(cnzs).evaluate()
    plt.plot(x, y, label='KDE /w silverman')
    plt.show()
    x, y = FFTKDE(kernel='gaussian', bw='ISJ').fit(cnzs).evaluate()
    plt.plot(x, y, label='KDE /w ISJ')
    plt.show()








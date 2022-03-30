"""
For Fig.2, drawing the influence of peak picking threshold pth.
"""
import os
import numpy as np
from mfe.from_txt import msi_from_txt, search_peak_th
import matplotlib.pyplot as plt


from settings import plot_params, path

plt.rcParams.update(plot_params)


def main(raw0, raw1, output):
    spectra0 = msi_from_txt(raw0)
    spectra1 = msi_from_txt(raw1)
    for key in list(spectra1.keys()):
        spectra0[(-key[0], -key[1])] = spectra1[key]
    ref_candidates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9, 0.95]
    metrics = search_peak_th(spectra0, ref_candidates)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('$p_{th}$')
    ax1.set_ylabel('Number of reference peaks')
    ax1.plot(ref_candidates, metrics['n_ref'], label='ref', color='black', marker='o', markersize=5)
    ax2 = ax1.twinx()
    ax2.plot(ref_candidates, metrics['tic_coverage'], label='int%', marker='o', markersize=5)
    ax2.plot(ref_candidates, np.abs(metrics['mean_error']), label='drft', marker='o', markersize=5)
    ax2.plot(ref_candidates, metrics['sparsity'], label='spar', marker='o', markersize=5)
    ax2.set_ylabel('Metrics')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.6),
               ncol=1, fancybox=True, shadow=True)
    plt.savefig(output, format='svg')
    plt.show()


if __name__ == "__main__":
    main(os.path.join(path['raw'],'SBB0-5cm_mz520-580.txt'), os.path.join(path['raw'],'SBB5-10cm_mz520-580.txt'), os.path.join(path['img'], 'a_peak_picking_threshold.svg'))
    main(os.path.join(path['raw'],'SBB0-5cm_mz375-525.txt'), os.path.join(path['raw'],'SBB5-10cm_mz375-525.txt'), os.path.join(path['img'], 'b_peak_picking_threshold.svg'))

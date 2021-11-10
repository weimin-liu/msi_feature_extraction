from mfe import create_feature_table
from mfe.src.util.calibration import suggest_calibrates, get_calibrate_err
from mfe.src.util.parse_txt import msi_from_txt
from mfe.src.vis.tic import get_tic_map
import numpy as np
import matplotlib.pyplot as plt
import concurrent
import tqdm

if __name__ == "__main__":
    raw_txt_path = r"../examples/SBB5-10cm_mz520-580.txt"
    msi = msi_from_txt(raw_txt_path)

    im = get_tic_map(msi)
    plt.imshow(im, cmap='viridis', vmin=np.nanquantile(im, 0.05), vmax=np.nanquantile(im, 0.95),
               interpolation='None')
    plt.axis('off')
    plt.tight_layout()
    plt.show(dpi=300)

    spot, mzs, result_arr = create_feature_table(519, 581, 0.01, msi)

    mzs_density = np.count_nonzero(result_arr, axis=0) / len(msi)

    mask = mzs_density >= 0.1

    mzs = mzs[mask]
    result_arr = result_arr[:, mask]

    mzs_density = np.count_nonzero(result_arr, axis=0) / len(msi)

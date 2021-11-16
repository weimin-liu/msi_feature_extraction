import pandas as pd

from mfe import create_feature_table
from mfe.src.util.parse_txt import msi_from_txt
import numpy as np

if __name__ == "__main__":
    raw_txt_path = r"../examples/SBB5-10cm_mz520-580.txt"
    msi = msi_from_txt(raw_txt_path)

    # im = get_tic_map(msi)
    # plt.imshow(im, cmap='viridis', vmin=np.nanquantile(im, 0.05), vmax=np.nanquantile(im, 0.95),
    #            interpolation='None')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show(dpi=300)

    spot, mzs, result_arr = create_feature_table(519, 581, 0.01, msi)

    mzs_density = np.count_nonzero(result_arr, axis=0) / len(msi)

    mask = mzs_density >= 0.1

    mzs = mzs[mask]
    result_arr = result_arr[:, mask]

    mzs_density = np.count_nonzero(result_arr, axis=0) / len(msi)

    feature_table = pd.DataFrame(result_arr, columns=mzs)

    feature_table['x'], feature_table['y'] = spot[:, 0], spot[:, 1]

    feature_table.to_csv(r'../examples/SBB5-10cm_mz520-580.csv')
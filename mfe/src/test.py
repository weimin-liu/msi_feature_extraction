from mfe.src.from_txt import msi_from_txt, get_ref_peaks, create_feature_table
import numpy as np

from mfe.src.peak_picking import get_peak_ranks

if __name__ == "__main__":
    raw_txt_path = r'../../examples/SBB5-10cm_mz520-580.txt'

    spectra = msi_from_txt(raw_txt_path)

    ref = get_ref_peaks(spectra)

    feature_table = create_feature_table(spectra, ref)

    t_df, ims = get_peak_ranks(feature_table)


"""Calibration algorithm for maldi imaging data"""

# Author: Weimin Liu wliu@marum.de

import concurrent
from bisect import bisect

import pandas as pd
import tqdm
import numpy as np
from .from_txt import msi_from_txt, create_feature_table, Spectrum


def quadric_calibration(spectrum, mzs, tol=10):
    source_mzs = spectrum.mz_values
    calibrate_dict = dict()
    for mz in mzs:
        idx = bisect(source_mzs, mz)
        if abs(source_mzs[idx - 1] - mz) < abs(source_mzs[idx] - mz):
            measured_mz = source_mzs[idx - 1]
        else:
            measured_mz = source_mzs[idx]
        err = mz - measured_mz
        if abs(err) / mz <= tol * 1e-6:
            calibrate_dict[measured_mz] = err
    if len(calibrate_dict) < 3:
        return None
    else:
        calibrate_df = pd.DataFrame.from_dict(calibrate_dict, orient='index')
        calibrate_df = calibrate_df.sort_index()
        z = np.polyfit(calibrate_df.index, calibrate_df.iloc[:, 0], 2)
        p = np.poly1d(z)
        test = p(spectrum.mz_values)
        new_mzs = spectrum._peaks_mz + test
        spectrum_new = Spectrum(new_mzs, spectrum.intensity_values,
                                mz_precision=spectrum.mz_precision, metadata=spectrum.metadata)
        return spectrum_new

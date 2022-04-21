"""Calibration algorithm for maldi imaging data"""

# Author: Weimin Liu wliu@marum.de

from bisect import bisect

import pandas as pd
import numpy as np
from .from_txt import Spectrum


def quadratic_calibration(spectrum, mzs: list, tol=10):
    """

    Args:
        spectrum:
        mzs: a list of mzs that are used for fitting quadratic calibration function
        tol: tolerance for finding the calibrates' peak

    Returns:

    """
    source_mzs = spectrum.mz_values
    calibrate_dict = {}
    for real_mz in mzs:
        idx = bisect(source_mzs, real_mz)
        if abs(source_mzs[idx - 1] - real_mz) < abs(source_mzs[idx] - real_mz):
            measured_mz = source_mzs[idx - 1]
        else:
            measured_mz = source_mzs[idx]
        err = real_mz - measured_mz
        if abs(err) / real_mz <= tol * 1e-6:
            calibrate_dict[measured_mz] = err
    if len(calibrate_dict) < 3:
        spectrum_new = None
    else:
        calibrate_df = pd.DataFrame.from_dict(calibrate_dict, orient='index')
        calibrate_df = calibrate_df.sort_index()
        z_func = np.polyfit(calibrate_df.index, calibrate_df.iloc[:, 0], 2)
        p_func = np.poly1d(z_func)
        test = p_func(spectrum.mz_values)
        new_mzs = spectrum.mz_values + test
        spectrum_new = Spectrum(new_mzs, spectrum.intensity_values,
                                mz_precision=spectrum.mz_precision, metadata=spectrum.metadata)
    return spectrum_new

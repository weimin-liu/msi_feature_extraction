"""Calibration algorithm for maldi imaging data"""

# Author: Weimin Liu wliu@marum.de

import concurrent
import pandas as pd
import tqdm
import numpy as np
from .from_txt import create_feature_table, msi_from_txt
from .depreciated.accurate_mz import get_accmz


# TODO:
#  - Add a class to perform lock mass calibration with multiple calibrates at the same time
#  - Add a class to calibrate time-based mass-to-charge ratio drift


def suggest_calibrates(raw_txt_path, bin_width=0.01):
    """
    Parameters:

        msi: a dictionary object containing the coordinate information with its corresponding spectrum

        bin_width: the width of mass bins in order to look for the most abudant compounds, default is 0.01

    Returns:

        acc_candidate_mzs: a list of mass-to-charge ratios widely occurred in the slide

        candidate_dist: kernel density estimated distribution of the mass-to-charge ratios of those candidates


    """

    msi = msi_from_txt(raw_txt_path)

    spot, mzs, result_arr = create_feature_table(bin_width, msi)

    result_arr = result_arr.toarray()

    mzs_density = np.count_nonzero(result_arr, axis=0) / len(msi)

    mzs_density = pd.DataFrame(np.stack((mzs, mzs_density)).T)

    mzs_density = mzs_density[mzs_density[1] >= 0.95]

    candidate_mzs = mzs_density[0].to_numpy()

    acc_candidate_mzs, candidate_dist = get_accmz(raw_txt_path, candidate_mzs)

    return acc_candidate_mzs, candidate_dist


def get_calibrate_err(calibrate_mz, spectrum, tol=20):
    """
    Parameters:

        calibrate_mz: mass-to-charge ratio of the calibrate to be evaluated

        spectrum: the uncalibrated spectrum to be evaluated

        tol: tolerance of the mass window in ppm

    Returns:

        err: drift of the calibrate in ppm
    """

    d = tol * calibrate_mz / 1e6
    drift = None
    mz_min = calibrate_mz - d
    mz_max = calibrate_mz + d
    mask = (spectrum.mz_values >= mz_min) & (spectrum.mz_values <= mz_max)
    mz_orig_arr = spectrum.mz_values[mask]
    tmp_int = spectrum.intensity_values[mask]
    if len(mz_orig_arr) > 0:
        mz_orig = mz_orig_arr[np.argmax(tmp_int)]
        drift = (calibrate_mz - mz_orig) * 1e6 / calibrate_mz
    return drift


class SimpleFallbackCalibrate:

    def __init__(self):
        """
        Calibrate with only one peak, supplemented by a fall-back list.
        """

        self.calibrates = None
        self.tol = None
        self.msi = None
        self.c_msi = dict()
        self.xy = None
        self.c_state = None
        self.c_params = None
        self.xy_dict = dict()

    def lock_mass(self, calibrate, xy):
        """
        Parameters
        --------
        xy: coordinates of the spectrum to be calibrated

        calibrate: index of the calibrate currently being used

        Returns
        --------
        c_found:
            - False: the calibrate is not found in the spectrum
            - integer: the calibrate is found in the spectrum, and also indicate the index of hit calibrate
        """
        drift = None
        d = self.tol * calibrate / 1e6
        key, spectrum = self.msi[xy]
        mz_min = calibrate - d
        mz_max = calibrate + d
        mask = (spectrum.mz_values >= mz_min) & (spectrum.mz_values <= mz_max)
        mz_orig_arr = spectrum.mz_values[mask]
        tmp_int = spectrum.intensity_values[mask]
        if len(mz_orig_arr) > 0:
            mz_orig = mz_orig_arr[np.argmax(tmp_int)]
            drift = calibrate - mz_orig
        return xy, drift

    def fit(self, msi: dict, calibrates: list, tol=10):
        """
        Parameters
        --------
        msi: a dictionary-like object, with x y coordinates as keys and the corresponding spectrum as values

        calibrates: list of calibrate

        tol: calibration tolerance in ppm

        self.c_state:
        - -1: no calibrates can be found using the provided calibrates list
        - integer: index of the calibrate found using the ranked calibrates list

        self.c_params: drift of mass-to-charge ratios from the calibrate
        """
        self.calibrates = calibrates
        self.tol = tol
        self.msi = msi
        self.xy = list(self.msi.keys())
        self.xy_dict = {k: v for v, k in enumerate(self.xy)}
        self.c_state = [-1] * len(self.msi)
        self.c_params = np.zeros(len(self.c_state))
        for calibrate_idx in range(len(self.calibrates)):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                to_do = []
                xy_todo = [self.xy[i] for i in range(len(self.c_state)) if self.c_state[i] == -1]
                for xy in xy_todo:
                    future = executor.submit(self.lock_mass, [self.calibrates[calibrate_idx], xy])
                    to_do.append(future)
                done_iter = concurrent.futures.as_completed(to_do)
                done_iter = tqdm.tqdm(done_iter, total=len(xy_todo))
                for future in done_iter:
                    res = future.result()
                    if res[1] is not None:
                        self.c_state[self.xy_dict[res[0]]] = calibrate_idx
                        self.c_params[self.xy_dict[res[0]]] = res[1]

    def transform(self):
        """
        Returns:
        --------
            c_msi: a dictionary-like object containing calibrated maldi spectrometry imaging data,
                    with x y coordinates as keys and the corresponding spectrum as values, with those spectra missing calibrates
                    dropped.
        """
        for i in range(len(self.xy)):
            if self.c_state[i] != -1:
                xy = self.xy[i]
                spectrum = self.msi[xy]
                drift = self.c_params[i]
                spectrum._peaks_mz += drift
                self.c_msi[xy] = spectrum
        return self.c_msi

    def fit_transform(self, msi: dict, calibrates: list, tol=10):
        self.fit(msi, calibrates, tol)
        self.transform()

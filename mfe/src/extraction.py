import math
import concurrent.futures
from collections import defaultdict
import tqdm
import numpy as np
import pandas as pd
from bisect import bisect
from scipy.sparse import vstack, csr_matrix

from mfe.src.util.Spectrum import Spectrum


def combine_spectrum(spot, spectrum, primer_df):
    spectrum_df = pd.DataFrame(spectrum.intensity_values, index=spectrum.mz_values)
    df = primer_df.combine_first(spectrum_df)
    df = df.replace(np.nan, 0)
    return spot, csr_matrix(df.to_numpy().flatten())


def binize(key, spectrum, mzbin, mzbin_precision):
    new_peaks = defaultdict(int)
    for mz in spectrum.mz_values:
        idx = bisect(mzbin, mz)
        if abs(mzbin[idx - 1] - mz) < abs(mzbin[idx] - mz):
            new_mz = mzbin[idx - 1]
        else:
            new_mz = mzbin[idx]
        # new_peaks[new_mz] += spectrum.intensity_at(mz)
        new_peaks[new_mz] = max(spectrum.intensity_at(mz), new_peaks[new_mz])
    spectrum_bin = Spectrum(list(new_peaks.keys()), list(new_peaks.values()), mz_precision=mzbin_precision,
                            metadata=spectrum.metadata)
    return key, spectrum_bin


def create_feature_table(mz_min: int, mz_max: int, size: float, spectrum_dict: dict):

    def _safe_arange(start, stop, step):
        return step * np.arange(start / step, stop / step)

    def mp_wrapper(func, source_dict, *args):
        to_do = list()
        for key in source_dict:
            future = executor.submit(func, key, source_dict[key], *args)
            to_do.append(future)
        done_iter = concurrent.futures.as_completed(to_do)
        done_iter = tqdm.tqdm(done_iter, total=len(to_do))
        target_dict = dict()
        for future in done_iter:
            res = future.result()
            target_dict[res[0]] = res[1]
        return target_dict

    mzs = _safe_arange(mz_min, mz_max, size)
    mz_bin_precision = int(-1 * math.log10(size))
    mzs = np.round(mzs, mz_bin_precision)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print("Binning the spectrum...")
        bin_spectrum_dict = mp_wrapper(binize, spectrum_dict, mzs, mz_bin_precision)
        print("Combining the binned spectrum...")
        primer_df = pd.DataFrame(np.zeros(mzs.shape), index=mzs)
        primer_df = primer_df.replace(0, np.nan)
        combined_spectrum_dict = mp_wrapper(combine_spectrum, bin_spectrum_dict, primer_df)
    spot = list(combined_spectrum_dict.keys())
    spot = np.array(spot)
    intensity = list(combined_spectrum_dict.values())
    result_arr = vstack(intensity)

    result_arr = result_arr.toarray()

    return spot, mzs, result_arr



import math
import concurrent.futures
import re
from collections import defaultdict
import tqdm
import numpy as np
import pandas as pd
from bisect import bisect

from KDEpy import FFTKDE
from scipy import signal
from scipy.sparse import vstack, csr_matrix

from mfe.src.util.Spectrum import Spectrum

# precision of mass-to-charge ratio to use before binning
MZ_PRECISION = 4


def parse_da_export(line: str, str_x=None, str_y=None):
    """
    parse lines in the plain text file exported from Bruker Data Analysis. Format of the plain text file is as follows:
    each line corresponds to one spot on the slide, with its x,y coordinate and the spectrum.

    Parameters:
    --------

        line: a single line in the plain text file, i.e., a single spot

        str_x: the regex string for how to extract the x coordinates

        str_y: the regex string for how to extract the y coordinates

    Returns:
    --------

        linked array in the form of [x-axis, y-axis, spectrum]
    """

    if (str_x is None) & (str_y is None):
        str_x = r'R00X(.*?)Y'

        str_y = r'Y(.*?)$'

    spot_name = line.split(";")[0]

    value = np.array(line.split(";")[2:]).reshape(-1, 3)

    mz = value[:, 0].astype(float)

    intensity = value[:, 1].astype(float)

    spectrum = Spectrum(mz, intensity, mz_precision=MZ_PRECISION, metadata=spot_name)

    x = re.findall(str_x, spot_name)[0]

    x = int(x)

    y = re.findall(str_y, spot_name)[0]

    y = int(y)

    return (x, y), spectrum


def msi_from_txt(raw_txt_path: str) -> dict:
    """
    convert the plain text file exported from Bruker DataAnalysis to a dictionary object, with x,y as the key and
    spectrum as the value

    Parameters:
    --------

        txt_file_path: plain text file exported from Bruker DA software

    Returns:
    -------

        A dictionary with [x, y] as keys and the corresponding spectrum as values
    """
    with open(raw_txt_path) as f:

        lines = f.readlines()

    lines = lines[1:]

    with concurrent.futures.ProcessPoolExecutor() as executor:

        to_do = []

        for line in lines:
            future = executor.submit(parse_da_export, line)

            to_do.append(future)

        done_iter = concurrent.futures.as_completed(to_do)

        done_iter = tqdm.tqdm(done_iter, total=len(lines))

        results = dict()

        for future in done_iter:
            res = future.result()

            results[res[0]] = res[1]

    return results


# TODO: replace the binning method with more advanced ones, such as
#  https://cran.r-project.org/web/packages/MALDIquant/MALDIquant.pdf
def combine_spectrum(spot: list, spectrum: Spectrum, primer_df: pd.DataFrame):
    """
    align the spectrum into mass bins

    Parameters:
    --------
        spot: a list object with [x, y] coordinate

        spectrum: a Spectrum object with the spectrum at the corresponding spot

        primer_df: a DataFrame object, with mass bins as index and spots as columns

    Returns:
    --------
        spot: a list object with [x, y] coordinate

        csr_matrix(df.to_numpy().flatten()): a sparse matrix with binned mass spectrum

    """

    spectrum_df = pd.DataFrame(spectrum.intensity_values, index=spectrum.mz_values)

    df = primer_df.combine_first(spectrum_df)

    df = df.replace(np.nan, 0)

    return spot, csr_matrix(df.to_numpy().flatten())


def binize(spot, spectrum: Spectrum, ref_peaks, tol=10):
    """
    function to bin a spectrum, the maximum peak in the binned area is used

    Parameters:
    --------
        spot: a list object with [x, y] coordinate

        spectrum: a Spectrum object with the spectrum at the corresponding spot

        mzbin: an array storing the m/z values of mass bins

        mzbin_precision: size of mass bins in Da

    Returns:

        spot: a list object with [x, y] coordinate

        spectrum_bin: a Spectrum object with the binned spectrum at the corresponding spot

    """
    new_peaks = defaultdict(int)

    for mz in spectrum.mz_values:

        if mz < np.max(ref_peaks):

            idx = bisect(ref_peaks, mz)

            if abs(ref_peaks[idx - 1] - mz) < abs(ref_peaks[idx] - mz):

                new_mz = ref_peaks[idx - 1]

            else:

                new_mz = ref_peaks[idx]

        else:
            new_mz = ref_peaks[-1]

        if abs(new_mz - mz) / new_mz <= tol * 1e-6:
            # new_peaks[new_mz] += spectrum.intensity_at(mz)
            new_peaks[new_mz] = max(spectrum.intensity_at(mz), new_peaks[new_mz])

    spectrum_bin = Spectrum(list(new_peaks.keys()), list(new_peaks.values()), metadata=spectrum.metadata)

    return spot, spectrum_bin


def get_ref_peaks(spectrum_dict: dict):
    """
    walk through all spectrum and find reference peaks for peak bining using Kernel Density Estimation

    Parameters:
    --------
        spectrum_dict:

    Returns:
    --------

    """

    # get all mzs from the sample and sort them
    mzs_all = [spec._peaks_mz for spec in spectrum_dict.values()]

    mzs_all = np.concatenate(mzs_all).ravel()

    mzs_all = np.sort(mzs_all)

    cluster = list()

    min_mz, max_mz = np.min(mzs_all), np.max(mzs_all)

    min_mz = int(round(min_mz, 0))

    max_mz = int(round(max_mz, 0))

    mz_bin = range(min_mz, max_mz)

    for i in tqdm.tqdm(range(len(mz_bin) - 1)):
        left = np.searchsorted(mzs_all, mz_bin[i])

        right = np.searchsorted(mzs_all, mz_bin[i + 1])

        cluster.append(mzs_all[left:right])

    ref_peaks = []

    for c in cluster:
        x, y = FFTKDE(kernel='gaussian', bw='ISJ').fit(c).evaluate()

        # TODO: add smoothing before peak detection
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        peak_th = 0.3

        peaks, _ = signal.find_peaks(y, height=peak_th)

        ref_peaks.extend([round(x[i], 4) for i in peaks])

    ref_peaks = np.sort(ref_peaks)

    return ref_peaks


def create_feature_table(spectrum_dict: dict, ref_peaks) -> pd.DataFrame:
    """
    create binned feature table with designated bin size
    Parameters:
    --------
        ref_peaks: a list of reference peak to which the samples aligned

        spectrum_dict: a dictionary object with key as spot coordinates and spectrum as value

    Returns:
    --------
        feature_table: a dataframe object

    """

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

    with concurrent.futures.ProcessPoolExecutor() as executor:

        print("Binning the spectrum...")

        bin_spectrum_dict = mp_wrapper(binize, spectrum_dict, ref_peaks)

        print("Combining the binned spectrum...")

        primer_df = pd.DataFrame(np.zeros(ref_peaks.shape), index=list(ref_peaks))

        primer_df = primer_df.replace(0, np.nan)

        combined_spectrum_dict = mp_wrapper(combine_spectrum, bin_spectrum_dict, primer_df)

    spot = list(combined_spectrum_dict.keys())

    spot = np.array(spot)

    intensity = list(combined_spectrum_dict.values())

    result_arr = vstack(intensity)

    result_arr = result_arr.toarray()

    feature_table = pd.DataFrame(result_arr, columns=list(ref_peaks))

    feature_table['x'], feature_table['y'] = spot[:, 0], spot[:, 1]

    return feature_table

"""
the main module to tackle plain text files exported from Data Analysis
"""
import concurrent.futures
import math
import re
import typing
from bisect import bisect
from collections import defaultdict, OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import tqdm
from KDEpy import FFTKDE
from matplotlib import pyplot as plt
from scipy import signal
from scipy.sparse import csr_matrix, vstack

# precision of mass-to-charge ratio to use before binning
MZ_PRECISION = 4


def parse_da_export(line: str, str_x=None, str_y=None):
    """
    parse lines in the plain text file exported from Bruker Data Analysis. Format of the plain
    text file is as follows: each line corresponds to one spot on the slide, with its x,
    y coordinate and the spectrum.

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

    mz_val = value[:, 0].astype(float)

    intensity = value[:, 1].astype(float)

    spectrum = Spectrum(mz_val, intensity, mz_precision=MZ_PRECISION, metadata=spot_name)

    x_loc = re.findall(str_x, spot_name)[0]

    x_loc = int(x_loc)

    y_loc = re.findall(str_y, spot_name)[0]

    y_loc = int(y_loc)

    return (x_loc, y_loc), spectrum


def msi_from_txt(raw_txt_path: str) -> dict:
    """
    convert the plain text file exported from Bruker DataAnalysis to a dictionary object, with x,
    y as the key and spectrum as the value

    Parameters:
    --------

        txt_file_path: plain text file exported from Bruker DA software

    Returns:
    -------

        A dictionary with [x, y] as keys and the corresponding spectrum as values
    """
    with open(raw_txt_path, encoding="utf8") as raw_txt:

        lines = raw_txt.readlines()

    lines = lines[1:]

    with concurrent.futures.ProcessPoolExecutor() as executor:

        to_do = []

        for line in lines:
            future = executor.submit(parse_da_export, line)

            to_do.append(future)

        done_iter = concurrent.futures.as_completed(to_do)

        done_iter = tqdm.tqdm(done_iter, total=len(lines))

        results = {}

        for future in done_iter:
            res = future.result()

            results[res[0]] = res[1]

    return results


class Spectrum:
    """
    modified from https://raw.githubusercontent.com/francisbrochu/msvlm/master/msvlm/msspectrum
    /spectrum.py
    """

    params = {'backend': 'pdf',
              'figure.dpi': 300,
              'axes.labelsize': 10,
              'font.size': 10,
              'legend.fontsize': 8,
              'legend.frameon': True,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'serif',
              'axes.linewidth': 0.5,
              'xtick.major.size': 4,  # major tick size in points
              'xtick.minor.size': 2,  # minor tick size in points
              'xtick.direction': 'out',
              'ytick.major.size': 4,  # major tick size in points
              'ytick.minor.size': 2,  # minor tick size in points
              'ytick.direction': 'out',
              }
    plt.rcParams.update(params)

    def __init__(self, mz_values, intensity_values, mz_precision=5, metadata=None):
        self._peaks_mz = np.array([])
        self._peaks_intensity = np.array([])
        self._median_normalized_peaks_intensity = np.array([])
        self._peaks = {}
        self.mz_err = np.array([])
        self.metadata = metadata
        self._mz_precision = mz_precision  # in decimals e.g.: mz_precision=3 => 5.342

        if len(mz_values) != len(intensity_values):
            raise ValueError("The number of mz values must be equal to the number of intensity "
                             "values.")

        self.set_peaks(mz_values, intensity_values)

    def peaks(self):
        """

        Returns: return protected _peaks

        """
        return self._peaks

    @property
    def mz_values(self):
        """
        Note: Returned values are always sorted
        Returns: return protected _peaks_mz
        """
        return self._peaks_mz

    @property
    def n_intensity_values(self):
        """
        return median normalized peak intensities

        Returns:

        """
        return self._median_normalized_peaks_intensity

    @property
    def mz_precision(self):
        """
        return the mz precision of the spectrum class
        Returns:

        """
        return self._mz_precision

    @mz_precision.setter
    def mz_precision(self, new_precision):
        self._mz_precision = new_precision
        self.set_peaks(self.mz_values, self.intensity_values)

    @property
    def intensity_values(self):
        """
        return protected intensity values
        Returns:

        """
        return self._peaks_intensity

    @property
    def tic(self):
        """
        return protected TIC values
        Returns:

        """
        return np.sum(self._peaks_intensity)

    def intensity_at(self, mz_val):
        """
        return the peak intensity at an exact mz value
        Args:
            mz_val:

        Returns:

        """
        mz_val = round(mz_val, self._mz_precision)
        try:
            intensity = self._peaks[mz_val]
        except AttributeError:
            intensity = 0.0
        return intensity

    def peak_around(self, mz_val, tol=10):
        """
        return the mz value and the intensity of closest peak to the mz_val
        Args:
            mz_val:
            tol:

        Returns:

        """
        mz_bin = tol * mz_val * (10 ** -6)
        mz_min = mz_val - mz_bin
        mz_max = mz_val + mz_bin
        mz_value_arr = self.mz_values[
            np.where((self.mz_values >= mz_min) & (self.mz_values <= mz_max))]
        if len(mz_value_arr) == 0:
            mz_value = None
            intensity_value = None
        else:
            intensity_arr = self.intensity_values[
                np.where((self.mz_values >= mz_min) & (self.mz_values <= mz_max))]
            intensity_value = np.max(intensity_arr)
            mz_value = mz_value_arr[np.argmax(intensity_arr)]
        return mz_value, intensity_value

    def intensity_around(self, mz_val, tol=10):
        """
        return the intensity of closest peak to the mz_val
        Args:
            mz_val:
            tol:

        Returns:

        """
        mz_bin = tol * mz_val * (10 ** -6)
        mz_min = mz_val - mz_bin
        mz_max = mz_val + mz_bin
        intensity_arr = self.intensity_values[
            np.where((self.mz_values >= mz_min) & (self.mz_values <= mz_max))]
        intensity = sum(intensity_arr)
        return intensity

    def set_peaks(self, mz_values, intensity_values):
        """
        set peaks using mzs and peak intensities
        Args:
            mz_values:
            intensity_values:

        Returns:

        """
        # This function must create a copy of mz_values and intensity_values to prevent the
        # modification of referenced arrays. This is assumed by other functions. Be careful!

        # Sort the peaks by mz
        sort_mz = np.argsort(mz_values)
        mz_values = np.asarray(mz_values)[sort_mz]
        intensity_values = np.asarray(intensity_values)[sort_mz]

        # Round the mz values based on the mz precision
        mz_values = np.asarray(np.round(mz_values, self._mz_precision), dtype=np.float)

        # Contiguous mz values might now be equivalent. Combine their intensity values by taking
        # the sum.
        unique_mz = np.unique(mz_values)
        unique_mz_intensities = np.zeros(unique_mz.shape)

        # Note: This assumes that mz_values and unique_mz are sorted
        if len(mz_values) != len(unique_mz):
            acc = 0
            current_unique_mz_idx = 0
            current_unique_mz = unique_mz[0]
            for i, mz_val in enumerate(mz_values):
                if mz_val != current_unique_mz:
                    unique_mz_intensities[current_unique_mz_idx] = acc  # Flush the accumulator
                    acc = 0  # Reset the accumulator
                    current_unique_mz_idx += 1  # Go to the next unique mz value
                    current_unique_mz = unique_mz[current_unique_mz_idx]  # Get the unique mz value
                acc += intensity_values[i]  # Increment the accumulator
            unique_mz_intensities[current_unique_mz_idx] = acc  # Flush the accumulator
        else:
            unique_mz_intensities = intensity_values

        self._peaks_mz = unique_mz
        self._peaks_intensity = unique_mz_intensities
        self._median_normalized_peaks_intensity = self._peaks_intensity / np.median(
            self._peaks_intensity)
        self._peaks = dict([(round(self._peaks_mz[i], self._mz_precision), self._peaks_intensity[i]) for i in
             range(len(self._peaks_mz))])

        self._check_peaks_integrity()

    def copy(self):
        """
        return a copy of the class
        Returns:

        """
        return deepcopy(self)

    def __repr__(self):
        return self.metadata

    def __iter__(self):
        """
        Returns an iterator on the peaks of the spectrum.

        Returns:
        --------
        peak_iterator: iterator
            An iterator that yields tuples of (mz, int) for each peaks in the spectrum.
        """
        return zip(self._peaks_mz, self._peaks_intensity)

    def __len__(self):
        """
        return the number of peaks in the spectrum
        Returns:

        """
        return self._peaks_mz.shape[0]

    def between(self, start, end, inplace=False):
        """
        return the spectrum between two mz values
        Args:
            start:
            end:
            inplace:

        Returns:

        """
        spec = self if inplace else self.copy()
        mz_values = spec.mz_values[np.where((spec.mz_values >= start) & (spec.mz_values <= end))]
        intensity_values = spec.intensity_values[
            np.where((spec.mz_values >= start) & (spec.mz_values <= end))]
        if inplace:
            spec.set_peaks(mz_values, intensity_values)
            tmp = None
        else:
            tmp = Spectrum(mz_values, intensity_values, mz_precision=spec.mz_precision,
                           metadata=spec.metadata)
            for key in spec.__dict__.keys():
                type_ = type(spec.__getattribute__(key)).__name__
                if type_ not in ['ndarray', 'list']:
                    tmp.__dict__[key] = spec.__dict__[key]
                else:
                    pass
        return tmp

    def show(self):
        """
        show spectrum
        Returns:

        """
        _, axis = plt.subplots()
        for i in range(len(self)):
            axis.plot([self.mz_values[i], self.mz_values[i]],
                    [0, 100 * self.intensity_values[i] / np.max(self.
                                                                intensity_values)], linewidth=0.3,
                    color='black')
        interval = (np.max(self.mz_values) - np.min(self.mz_values)) / 30
        min_mz = max(0, math.floor(self.mz_values[0] / interval - 1) * interval)
        max_mz = max(0, math.floor(self.mz_values[-1] / interval + 1) * interval)
        axis.set_xlim(min_mz, max_mz)
        # remove top and right axis
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()
        # label axes
        axis.set_xlabel(r"m/z")
        axis.set_ylabel(r"Intensity")
        # set x labels
        start, end = axis.get_xlim()
        axis.set_xticks(np.linspace(start, end + 1, 10))
        # set y labels
        axis.set_ylim(0, 100)
        start, end = axis.get_ylim()
        axis.set_yticks(np.arange(start, end + 1, 10))
        axis.grid(True, axis="y", color='black', linestyle=':', linewidth=0.1)
        return axis

    def _check_peaks_integrity(self):
        if not len(self._peaks_mz) == len(self._peaks_intensity):
            raise ValueError(
                "The number of mz values must be equal to the number of intensity values.")
        if not all(
                self._peaks_mz[i] <= self._peaks_mz[i + 1] for i in range(len(self._peaks_mz) - 1)):
            raise ValueError("Mz values must be sorted.")
        if len(np.unique(self._peaks_mz)) != len(self._peaks_mz):
            raise ValueError("Mz value list contains duplicate values.")


def combine_spectrum(spot: list, spectrum: Spectrum, primer_df: pd.DataFrame, normalization='None'):
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

        csr_matrix(err_df.to_numpy().flatten()): a sparse matrix with the mass error before and
        after peak alignment

    """
    if normalization == 'None':
        spectrum_df = pd.DataFrame(spectrum.intensity_values, index=spectrum.mz_values)
    elif normalization == 'median':
        spectrum_df = pd.DataFrame(spectrum.n_intensity_values, index=spectrum.mz_values)
    else:
        raise NotImplementedError

    err_df = pd.DataFrame(spectrum.mz_err, index=spectrum.mz_values)

    comb_df = primer_df.combine_first(spectrum_df)

    err_df = primer_df.combine_first(err_df)

    comb_df = comb_df.replace(np.nan, 0)

    return spot, csr_matrix(comb_df.to_numpy().flatten()), csr_matrix(err_df.to_numpy().flatten())


def create_bin(spot, spectrum: Spectrum, ref_peaks, tol=10):
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

    mz_err = {}

    for mz_val in spectrum.mz_values:

        if mz_val < np.max(ref_peaks):

            idx = bisect(ref_peaks, mz_val)

            if abs(ref_peaks[idx - 1] - mz_val) < abs(ref_peaks[idx] - mz_val):

                new_mz = ref_peaks[idx - 1]

            else:

                new_mz = ref_peaks[idx]

        else:
            new_mz = ref_peaks[-1]

        err = (new_mz - mz_val) / new_mz

        if abs(err) <= tol * 1e-6:
            # new_peaks[new_mz] += spectrum.intensity_at(mz)
            new_peaks[new_mz] = max(spectrum.intensity_at(mz_val), new_peaks[new_mz])
            mz_err[new_mz] = err * 1e6

    spectrum_bin = Spectrum(list(new_peaks.keys()), list(new_peaks.values()),
                            metadata=spectrum.metadata)

    mz_err = OrderedDict(sorted(mz_err.items()))

    spectrum_bin.mz_err = np.array(list(mz_err.values()))

    return spot, spectrum_bin


def create_feature_table(spectrum_dict: dict, ref_peaks, tol=10, normalization='None') -> \
        typing.Tuple[
            pd.DataFrame, pd.DataFrame]:
    """
    create binned feature table with designated bin size
    Parameters:
    --------
        ref_peaks: a list of reference peak to which the samples aligned

        spectrum_dict: a dictionary object with key as spot coordinates and spectrum as value

        tol

    Returns:
    --------
        feature_table: a dataframe object

        err_table: return the mz error of each aligned peak in each spot for accuracy evaluation

    """

    def mp_wrapper(func, source_dict, *args, **kwargs):
        to_do = []
        for key in source_dict:
            future = executor.submit(func, key, source_dict[key], *args, **kwargs)
            to_do.append(future)
        done_iter = concurrent.futures.as_completed(to_do)
        done_iter = tqdm.tqdm(done_iter, total=len(to_do))
        target_dict = {}
        for future in done_iter:
            res = future.result()
            target_dict[res[0]] = res
        return target_dict

    with concurrent.futures.ProcessPoolExecutor() as executor:

        print("Binning the spectrum...")

        bin_spectrum_dict = mp_wrapper(create_bin, spectrum_dict, ref_peaks, tol)

        bin_spectrum_dict = {key: value[1] for key, value in bin_spectrum_dict.items()}

        print("Combining the binned spectrum...")

        primer_df = pd.DataFrame(np.zeros(ref_peaks.shape), index=list(ref_peaks))

        primer_df = primer_df.replace(0, np.nan)

        combined_spectrum_dict = mp_wrapper(combine_spectrum, bin_spectrum_dict, primer_df,
                                            normalization=normalization)

        err_dict = {key: value[2] for key, value in combined_spectrum_dict.items()}

        combined_spectrum_dict = {key: value[1] for key, value in
                                  combined_spectrum_dict.items()}

    spot = list(combined_spectrum_dict.keys())

    spot = np.array(spot)

    result_arr = vstack(list(combined_spectrum_dict.values()))

    err_arr = vstack(list(err_dict.values()))

    result_arr = result_arr.toarray()

    err_arr = err_arr.toarray()

    feature_table = pd.DataFrame(result_arr, columns=list(ref_peaks))

    err_table = pd.DataFrame(err_arr, columns=list(ref_peaks))

    feature_table['x'], feature_table['y'] = spot[:, 0], spot[:, 1]

    err_table['x'], err_table['y'] = spot[:, 0], spot[:, 1]

    return feature_table, err_table


def find_peaks(y_val, peak_picking_method=None, peak_th=None):
    """
    find peaks
    Args:
        y_val:
        peak_picking_method:
        peak_th:

    Returns:

    """
    if peak_picking_method == 'height':
        peaks, _ = signal.find_peaks(y_val, height=peak_th)
    elif peak_picking_method == 'prominence':
        peaks, _ = signal.find_peaks(y_val, prominence=(peak_th, None))
    else:
        raise NotImplementedError(
            "The peak picking method chosen hasn't been implemented yet!")
    return peaks


def find_peaks_in_bin(peak_th=None, ref_peaks=None, c_dist=None, peak_picking_method=None):
    """

    Args:
        peak_th:
        ref_peaks:
        c_dist:
        peak_picking_method:

    Returns:

    """
    x_val, y_val = FFTKDE(kernel='gaussian', bw='ISJ').fit(c_dist).evaluate()

    y_val = (y_val - np.min(y_val)) / (np.max(y_val) - np.min(y_val))

    if isinstance(peak_th, float):
        peaks = find_peaks(y_val, peak_picking_method=peak_picking_method, peak_th=peak_th)
        ref_peaks[peak_th].extend([round(x_val[i], 4) for i in peaks])

    elif isinstance(peak_th, list):
        for p_th in peak_th:
            peaks = find_peaks(y_val, peak_picking_method=peak_picking_method, peak_th=p_th)
            ref_peaks[p_th].extend([round(x_val[i], 4) for i in peaks])
    else:
        raise NotImplementedError
    return ref_peaks


def get_ref_peaks(spectrum_dict: dict or list, peak_picking_method='prominence', peak_th=0.1):
    """
    walk through all spectrum and find reference peaks for peak bining using Kernel Density
    Estimation

    Parameters:
    --------
        spectrum_dict:

        peak_picking_method: 'prominence' or 'height', used in function scipy.signal.find_peaks.

        peak_th: the threshold for distinguishing between peaks and noises, used in the function
        scipy.signal.find_peaks. The value should be between [0, 1), the higher the value,
        more peaks will be picked here. if the peak_picking_method is set to 'height',
        then scipy.signal.find_peaks(, height=peak_th), if is set to 'prominence',
        then scipy.signal.find_peaks(, prominence=(peak_th, None]). Read
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html for
        more reference.

    Returns:
    --------

    """
    mzs_all = []
    # get all mzs from the sample and sort them
    if isinstance(spectrum_dict, dict):
        mzs_all = [spec.mz_values for spec in spectrum_dict.values()]
    elif isinstance(spectrum_dict, list):
        for spectrum_dict_item in spectrum_dict:
            mzs_all.extend([spec.mz_values for spec in spectrum_dict_item.values()])

    mzs_all = np.concatenate(mzs_all).ravel()

    mzs_all = np.sort(mzs_all)

    cluster = []

    mz_bin = range(int(round(np.min(mzs_all), 0)), int(round(np.max(mzs_all), 0)))

    print(f'Detecting reference peaks with peak prominence greater than {peak_th}')
    for i in range(len(mz_bin) - 1):
        left = np.searchsorted(mzs_all, mz_bin[i])

        right = np.searchsorted(mzs_all, mz_bin[i + 1])

        cluster.append(mzs_all[left:right])

    ref_peaks = {}

    if isinstance(peak_th, float):

        ref_peaks[peak_th] = []

    elif isinstance(peak_th, list):

        for p_th in peak_th:
            ref_peaks[p_th] = []

    for c_dist in cluster:
        ref_peaks = find_peaks_in_bin(peak_th=peak_th, ref_peaks=ref_peaks, c_dist=c_dist,
                                      peak_picking_method=peak_picking_method)

    for key in list(ref_peaks.keys()):
        ref_peaks[key] = np.sort(ref_peaks[key])

    return ref_peaks


def search_peak_th(raw_data: dict, peak_th_candidates,
                   peak_picking_method='prominence') -> dict:
    """
    This toolbox function can be used to decide which peak_th parameter should be used to get
    reference peaks. It will return four metrics for consideration:

    - Number of reference peaks: the number of reference peaks discovered

    - TIC Coverage: how many percentage of the peak intensity is covered after peak alignment

    - Accuracy: the overall m/z error between the measured peaks and the reference peak,
    including mean and std

    - Sparsity: how sparse is the feature table that is obtained from the reference peaks

    Args:
        raw_data: the raw data in dictionary form, returned by msi_from_txt()

        peak_th_candidates: a list of candidates to be considered

        peak_picking_method: the method for peak alignment, default to 'prominent'

    Returns:

    """

    cover = []

    drft = []

    spar = []

    ref = get_ref_peaks(raw_data, peak_picking_method=peak_picking_method,
                        peak_th=peak_th_candidates)

    n_ref = list(ref.values())

    min_pth = np.min(peak_th_candidates)

    feature_table, err_table = create_feature_table(raw_data, ref[min_pth])

    for value in ref.values():
        feature_table_sub = feature_table[value]

        feature_table_sub[['x', 'y']] = feature_table[['x', 'y']]

        err_table_sub = err_table[value]

        err_table_sub[['x', 'y']] = err_table[['x', 'y']]

        coverage = peak_alignment_evaluation(raw_data, feature_table_sub)

        cover.append(coverage['TIC_coverage'].mean())

        drft.append(
            err_table_sub.drop(columns=['x', 'y']).abs().mean(skipna=True).mean(skipna=True))

        spar.append((feature_table_sub.drop(columns=['x', 'y']).to_numpy() == 0).mean())

    return {
        'n_ref': n_ref,
        'tic_coverage': cover,
        'mean_error': drft,
        'sparsity': spar
    }


def peak_alignment_evaluation(spectrum_dict: dict, feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Use this function to evaluate the performance of peak alignment. It has two metrics,
    the first is the coverage of the intensity of the aligned peaks, the second is the accuracy,
    which measures the discrepancy of the reference peaks comparing to the measured peaks before
    alignment.

    Args:
        spectrum_dict: the dictionary representing raw data

        feature_table: the feature table obtained after peak alignment

    Returns:

        coverage_dict: the percentage of tic that's been picked after alignment

    """

    if len(spectrum_dict) != len(feature_table):
        raise ValueError("This feature table is not derived from the current raw data!")

    feature = feature_table
    feature = feature.set_index(['x', 'y'])

    keys = list(spectrum_dict.keys())

    keys_df = pd.DataFrame(keys)

    keys_df = keys_df.set_index([0, 1])

    feature = feature.loc[keys_df.index, :]

    feature = np.array(feature)

    tic_after = np.sum(feature, axis=1)

    coverage_dict = {}

    m_idx = 0

    for key in spectrum_dict.keys():

        coverage_dict[key] = tic_after[m_idx] / spectrum_dict[key].tic

        if coverage_dict[key] > 1:
            raise ValueError(
                'Something went wrong! TIC after alignment should not be greater than before!')

        m_idx += 1

    coverage = pd.DataFrame([coverage_dict]).T

    tmp = pd.DataFrame(list(coverage.index))

    coverage = coverage.reset_index()

    coverage[['x', 'y']] = tmp

    coverage = coverage[['x', 'y', 0]].rename(columns={0: 'TIC_coverage'})

    return coverage

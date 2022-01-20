import math
import time
from copy import deepcopy
from datetime import datetime as dt

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

from src.mfe.from_txt import get_ref_peaks, create_feature_table


def imshow(array: np.arrary, fill=0) -> np.arrary:
    """

    Args:
        array: array with a shape of (,3) to be pivoted to an image.

        fill: values to fill missing values

    Returns:
        im: image matrix

    """
    if isinstance(array, np.ndarray):
        pass
    elif isinstance(array, pd.DataFrame):
        array = array.to_numpy()
    else:
        raise NotImplementedError('The data type is not implemented! Must be array or pandas dataframe!')

    xLocation = array[:, 0].astype(int)
    yLocation = array[:, 1].astype(int)
    xLocation = xLocation - min(xLocation)
    yLocation = yLocation - min(yLocation)
    col = max(np.unique(xLocation))
    row = max(np.unique(yLocation))
    im = np.zeros((col, row))
    im[:] = fill
    for i in range(len(xLocation)):
        im[np.asscalar(xLocation[i]) - 1, np.asscalar(yLocation[i] - 1)] = array[i, 2]
    return im


class Spectrum:
    """
    modified from https://raw.githubusercontent.com/francisbrochu/msvlm/master/msvlm/msspectrum/spectrum.py
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
        self._peaks = {}
        self._mz_err = np.array([])
        self._is_calibrated = False
        self.metadata = metadata
        self._mz_precision = mz_precision  # in decimals e.g.: mz_precision=3 => 5.342

        if len(mz_values) != len(intensity_values):
            raise ValueError("The number of mz values must be equal to the number of intensity values.")

        self.set_peaks(mz_values, intensity_values)

    def peaks(self):
        """
        Note: Peaks are not necessarily sorted here because of dict
        """
        return self._peaks

    @property
    def mz_values(self):
        """
        Note: Returned values are always sorted
        """
        return self._peaks_mz

    @property
    def mz_err(self):
        return self._mz_err

    @property
    def mz_precision(self):
        return self._mz_precision

    @mz_precision.setter
    def mz_precision(self, new_precision):
        self._mz_precision = new_precision
        self.set_peaks(self.mz_values, self.intensity_values)

    @property
    def intensity_values(self):
        return self._peaks_intensity

    @property
    def tic(self):
        return np.sum(self._peaks_intensity)

    def intensity_at(self, mz):
        mz = round(mz, self._mz_precision)
        try:
            intensity = self._peaks[mz]
        except AttributeError:
            intensity = 0.0
        return intensity

    def peak_around(self, mz, tol=10):
        mz_bin = tol * mz * (10 ** -6)
        mz_min = mz - mz_bin
        mz_max = mz + mz_bin
        mz_value_arr = self.mz_values[np.where((self.mz_values >= mz_min) & (self.mz_values <= mz_max))]
        if len(mz_value_arr) == 0:
            mz_value = None
            intensity_value = None
        else:
            intensity_arr = self.intensity_values[np.where((self.mz_values >= mz_min) & (self.mz_values <= mz_max))]
            intensity_value = np.max(intensity_arr)
            mz_value = mz_value_arr[np.argmax(intensity_arr)]
        return mz_value, intensity_value

    def intensity_around(self, mz, tol=10):
        mz_bin = tol * mz * (10 ** -6)
        mz_min = mz - mz_bin
        mz_max = mz + mz_bin
        intensity_arr = self.intensity_values[np.where((self.mz_values >= mz_min) & (self.mz_values <= mz_max))]
        intensity = sum(intensity_arr)
        return intensity

    def set_peaks(self, mz_values, intensity_values):
        # XXX: This function must create a copy of mz_values and intensity_values to prevent the modification of
        # referenced arrays. This is assumed by other functions. Be careful!

        # Sort the peaks by mz
        sort_mz = np.argsort(mz_values)
        mz_values = np.asarray(mz_values)[sort_mz]
        intensity_values = np.asarray(intensity_values)[sort_mz]

        # Round the mz values based on the mz precision
        mz_values = np.asarray(np.round(mz_values, self._mz_precision), dtype=np.float)

        # Contiguous mz values might now be equivalent. Combine their intensity values by taking the sum.
        unique_mz = np.unique(mz_values)
        unique_mz_intensities = np.zeros(unique_mz.shape)

        # Note: This assumes that mz_values and unique_mz are sorted
        if len(mz_values) != len(unique_mz):
            acc = 0
            current_unique_mz_idx = 0
            current_unique_mz = unique_mz[0]
            for i, mz in enumerate(mz_values):
                if mz != current_unique_mz:
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

        self._peaks = dict([(round(self._peaks_mz[i], self._mz_precision), self._peaks_intensity[i]) for i in
                            range(len(self._peaks_mz))])

        self._check_peaks_integrity()

    def copy(self):
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
        return self._peaks_mz.shape[0]

    def between(self, start, end, inplace=False):
        spec = self if inplace else self.copy()
        mz_values = spec.mz_values[np.where((spec.mz_values >= start) & (spec.mz_values <= end))]
        intensity_values = spec.intensity_values[
            np.where((spec.mz_values >= start) & (spec.mz_values <= end))]
        if not inplace:
            tmp = Spectrum(mz_values, intensity_values, mz_precision=spec.mz_precision, metadata=spec.metadata)
            for key in spec.__dict__.keys():
                type_ = type(spec.__getattribute__(key)).__name__
                if type_ not in ['ndarray', 'list']:
                    tmp.__dict__[key] = spec.__dict__[key]
                else:
                    pass
            return tmp
        else:
            spec.set_peaks(mz_values, intensity_values)

    def show(self):
        fig, ax = plt.subplots()
        for i in range(len(self)):
            ax.plot([self.mz_values[i], self.mz_values[i]],
                    [0, 100 * self.intensity_values[i] / np.max(self.
                                                                intensity_values)], linewidth=0.3, color='black')
        interval = (np.max(self.mz_values) - np.min(self.mz_values)) / 30
        min_mz = max(0, math.floor(self.mz_values[0] / interval - 1) * interval)
        max_mz = max(0, math.floor(self.mz_values[-1] / interval + 1) * interval)
        ax.set_xlim(min_mz, max_mz)
        # remove top and right axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # label axes
        ax.set_xlabel(r"m/z")
        ax.set_ylabel(r"Intensity")
        # set x labels
        start, end = ax.get_xlim()
        ax.set_xticks(np.linspace(start, end + 1, 10))
        # set y labels
        ax.set_ylim(0, 100)
        start, end = ax.get_ylim()
        ax.set_yticks(np.arange(start, end + 1, 10))
        ax.grid(True, axis="y", color='black', linestyle=':', linewidth=0.1)
        return ax

    def _check_peaks_integrity(self):
        if not len(self._peaks_mz) == len(self._peaks_intensity):
            raise ValueError("The number of mz values must be equal to the number of intensity values.")
        if not all(self._peaks_mz[i] <= self._peaks_mz[i + 1] for i in range(len(self._peaks_mz) - 1)):
            raise ValueError("Mz values must be sorted.")
        if len(np.unique(self._peaks_mz)) != len(self._peaks_mz):
            raise ValueError("Mz value list contains duplicate values.")


class CorSolver:
    """
    A class that takes tie points from two coordinate system (i.e., one from Flex, the other from xray measurement)
    and solve the transformation between these two systems.
    """

    def __init__(self):
        self.translation_vector = None
        self.transformation_matrix = None

    def _reset(self):
        if hasattr(self, 'transformation_matrix'):
            del self.transformation_matrix
            del self.translation_vector

    def fit(self, src_tri, dst_tri):
        """
        Parameters:
        --------
            src_tri: coordinates of the source triangle in the source coordinate system
            dst_tri:  coordinates of the target triangle in the target coordinate system
        """
        self._reset()
        return self.partial_fit(src_tri, dst_tri)

    def partial_fit(self, src_tri, dst_tri):
        """
        solve the affine transformation matrix between FlexImage coordinates and X_ray coordinates
        https://stackoverflow.com/questions/56166088/how-to-find-affine-transformation-matrix-between-two-sets-of-3d-points
        """
        l = len(src_tri)
        B = np.vstack([np.transpose(src_tri), np.ones(l)])
        D = 1.0 / np.linalg.det(B)

        def entry(r, d):
            return np.linalg.det(np.delete(np.vstack([r, B]), (d + 1), axis=0))

        M = [[(-1) ** i * D * entry(R, i) for i in range(l)] for R in np.transpose(dst_tri)]
        A, t = np.hsplit(np.array(M), [l - 1])
        t = np.transpose(t)[0]
        self.transformation_matrix = A
        self.translation_vector = t
        return self

    def transform(self, src_coordinate):
        """
        Parameters:
        --------
            src_coordinate: the source coordinates that needs be transformed
        """
        dst_coordinate = src_coordinate.dot(self.transformation_matrix.T) + self.translation_vector
        return np.round(dst_coordinate, 0)


def since_epoch(date):
    """
    return the time span in seconds since epoch
    """
    return time.mktime(date.timetuple())


def from_year_fraction(date: float):
    """
    this function takes a date in float as input and return datetime as output

    Parameter:
    --------
        date: float such as 2000.02

    Return:
    --------
        the corresponding datetime object
    """
    if type(date).__name__ == 'float':
        s = since_epoch
        year = int(date)
        fraction = date - year
        start_of_this_year = dt(year=year, month=1, day=1)
        start_of_next_year = dt(year=year + 1, month=1, day=1)
        year_duration = s(start_of_next_year) - s(start_of_this_year)
        year_elapsed = fraction * year_duration
        return dt.fromtimestamp(year_elapsed + s(start_of_this_year))
    else:
        raise TypeError


def to_year_fraction(date):
    """
    this function takes datetime object as input and return float date as output

    Parameter:
    --------
        date: a datetime object

    Return:
    --------
        the corresponding float of the datetime object
    """
    s = since_epoch
    year = date.year
    start_of_this_year = dt(year=year, month=1, day=1)
    start_of_next_year = dt(year=year + 1, month=1, day=1)
    year_elapsed = s(date) - s(start_of_this_year)
    year_duration = s(start_of_next_year) - s(start_of_this_year)
    fraction = year_elapsed / year_duration
    return date.year + fraction


def search_peak_th(raw_data: dict, peak_th_candidates: list, peak_picking_method='prominence') -> dict:
    """
    This toolbox function can be used to decide which peak_th parameter should be used to get reference peaks. It
    will return four metrics for consideration:

    - Number of reference peaks: the number of reference peaks discovered

    - TIC Coverage: how many percentage of the peak intensity is covered after peak alignment

    - Accuracy: the overall m/z error between the measured peaks and the reference peak, including mean and std

    - Sparsity: how sparse is the feature table that is obtained from the reference peaks

    Args:
        raw_data: the raw data in dictionary form, returned by msi_from_txt()

        peak_th_candidates: a list of candidates to be considered

        peak_picking_method: the method for peak alignment, default to 'prominent'

    Returns:

    """

    # TODO: add sanity check before processing

    n_ref = list()

    cover = list()

    me = list()

    mstd = list()

    spar = list()

    for peak_th in tqdm.tqdm(peak_th_candidates):

        ref = get_ref_peaks(raw_data, peak_picking_method=peak_picking_method, peak_th=peak_th)

        n_ref.append(len(ref))

        feature_table, err_table = create_feature_table(raw_data, ref)

        coverage = peak_alignment_evaluation(raw_data, feature_table)

        cover.append(coverage['TIC_coverage'].mean())

        me.append(err_table.drop(columns=['x', 'y']).mean(skipna=True).mean(skipna=True))

        mstd.append(err_table.drop(columns=['x', 'y']).std(skipna=True).mean(skipna=True))

        spar.append((feature_table.drop(columns=['x', 'y']).to_numpy() == 0).mean())

    return {
        'n_ref': n_ref,
        'tic_coverage': cover,
        'mean_error': me,
        'mean_std': mstd,
        'sparsity': spar
    }


def peak_alignment_evaluation(spectrum_dict: dict, feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Use this function to evaluate the performance of peak alignment. It has two metrics, the first is the coverage of
    the intensity of the aligned peaks, the second is the accuracy, which measures the discrepancy of the reference
    peaks comparing to the measured peaks before alignment.

    Args:
        spectrum_dict: the dictionary representing raw data

        feature_table: the feature table obtained after peak alignment

    Returns:

        coverage_dict: the percentage of tic that's been picked after alignment

    """

    if len(spectrum_dict) != len(feature_table):
        raise ValueError("This feature table is not derived from the current raw data!")
    else:

        feature = feature_table
        feature = feature.set_index(['x', 'y'])

        keys = list(spectrum_dict.keys())

        keys_df = pd.DataFrame(keys)

        keys_df = keys_df.set_index([0, 1])

        feature = feature.loc[keys_df.index, :]

        feature = np.array(feature)

        tic_after = np.sum(feature, axis=1)

        coverage_dict = {}

        for m in range(len(spectrum_dict)):

            key = list(spectrum_dict.keys())[m]

            coverage_dict[key] = tic_after[m] / spectrum_dict[key].tic

            if coverage_dict[key] > 1:

                raise ValueError('Something went wrong! TIC after alignment should not be greater than before!')

        coverage = pd.DataFrame([coverage_dict]).T

        tmp = pd.DataFrame(list(coverage.index))

        coverage = coverage.reset_index()

        coverage[['x', 'y']] = tmp

        coverage = coverage[['x', 'y', 0]].rename(columns={0: 'TIC_coverage'})

    return coverage
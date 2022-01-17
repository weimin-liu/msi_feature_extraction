import numpy as np
from copy import deepcopy
import math
import matplotlib.pyplot as plt

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


class Spectrum:
    """
    modified from https://raw.githubusercontent.com/francisbrochu/msvlm/master/msvlm/msspectrum/spectrum.py
    """

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

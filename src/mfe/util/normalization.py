import concurrent
import functools
import tqdm

import numpy as np

from src.mfe import Spectrum


def print_doc(func):
    """
    print the docstring of the decorated function
    """
    @functools.wraps(func)
    def wrapper_docstring(*args, **kwargs):
        print(func.__doc__)
        value = func(*args, **kwargs)
        return value

    return wrapper_docstring


def apply_tic_normalization(spot, spectrum):
    """
    the spot argument is just passed to be returned, no operation should be performed on that argument
    """
    mz = spectrum.mz_values
    tic = np.sum(spectrum.intensity_values)
    intensity = spectrum.intensity_values / tic
    spectrum_n = Spectrum(mz, intensity, mz_precision=spectrum.mz_precision, metadata=spectrum.metadata)
    return spot, spectrum_n


@print_doc
def tic_normalize(spectrum_dict: dict) -> dict:
    """
    Peak intensity of the spectrum are now being TIC-normalized. You should perform this function after
    the removal of outlier spots  based on TIC.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        to_do = list()
        for key in spectrum_dict:
            future = executor.submit(apply_tic_normalization, key, spectrum_dict[key])
            to_do.append(future)
        done_iter = concurrent.futures.as_completed(to_do)
        done_iter = tqdm.tqdm(done_iter, total=len(to_do))
        results = dict()
        for future in done_iter:
            res = future.result()
            results[res[0]] = res[1]
        return results
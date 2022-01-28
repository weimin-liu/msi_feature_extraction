import concurrent
import functools
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from KDEpy import FFTKDE


def parse_spot_name(spot):
    x = re.findall(r'R00X(.*?)Y', spot)[0]
    x = int(x)
    y = re.findall(r'Y(.*?)$', spot)[0]
    y = int(y)
    return x, y


def parse_line(line):
    spot = line.split(';')[0]
    arr = np.array(line.split(';')[2:]).reshape(-1, 3)
    arr = arr.astype(float)
    mzs = arr[:, 0].flatten()
    intensity = arr[:, 1].flatten()
    return spot, mzs, intensity


def extract_from_single_line(mz: float or list, bin_size, line):
    spot, mzs, intensity = parse_line(line)

    result = list()

    try:
        for m in mz:
            mask = (mzs >= (m - bin_size / 2)) & (mzs < (m + bin_size / 2))
            intensity0 = intensity[mask]
            mzs0 = mzs[mask]
            if len(intensity0) != 0:
                result.append([spot, mzs0[np.argmax(intensity0)].round(5), intensity0[np.argmax(intensity0)], m])
    except TypeError:
        mask = (mzs >= (mz - bin_size / 2)) & (mzs < (mz + bin_size / 2))
        intensity = intensity[mask]
        mzs = mzs[mask]
        if len(mzs) != 0:
            result.append([spot, mzs[np.argmax(intensity)].round(5), intensity[np.argmax(intensity)], mz])
    return result


def get_accmz(txt_path, mzs, bin_size=0.01):
    txt_path = txt_path
    candidate_mz = mzs

    with open(txt_path, 'r') as f:
        raw = f.readlines()

    f = functools.partial(extract_from_single_line, candidate_mz, bin_size)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        peaks = list(tqdm(executor.map(f, raw), total=len(raw)))

    peaks = [item[1:] for sublist in peaks for item in sublist]
    peaks = np.array(peaks)

    candidate_dict = dict()
    candidate_dist = dict()
    center_mzs = list()
    i = 0
    for mz in candidate_mz:
        candidate_dict[mz] = peaks[peaks[:, 2] == mz]
        x, y = FFTKDE(kernel="gaussian", bw=0.001).fit(candidate_dict[mz][:, 0]).evaluate()
        maxid = np.argmax(y)
        center_mz = x[maxid].round(4)
        center_mzs.append(center_mz)
        i += 1
        candidate_dist[mz] = np.stack((x, y))
    return dict(zip(mzs, center_mzs)), candidate_dist


def get_accurate_intensity(path, center_mzs, tol=0.004):
    with open(path, 'r') as f:
        raw = f.readlines()
    f = functools.partial(extract_from_single_line, center_mzs, tol)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        peaks = list(tqdm(executor.map(f, raw), total=len(raw)))

    peaks = [item for sublist in peaks for item in sublist]

    data = pd.DataFrame(peaks)
    data = pd.pivot_table(data, values=2, index=0, columns=3)
    return data

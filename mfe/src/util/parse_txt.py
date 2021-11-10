import concurrent
import re
import tqdm
import numpy as np

from mfe.src.util.Spectrum import Spectrum

# precision of mass-to-charge ratio to use before binning
MZ_PRECISION = 4


def parse_da_export(line: str):
    """
    parse lines in the plain text file exported from Bruker Data Analysis. Format of the plain text file is as follows:
    each line corresponds to one spot on the slide, with its x,y coordinate and the spectrum.

    Parameters:
    --------

        line: a single line in the plain text file, i.e., a single spot

    Returns:
    --------

        linked array in the form of [x-axis, y-axis, spectrum]
    """
    spot_name = line.split(";")[0]
    value = np.array(line.split(";")[2:]).reshape(-1, 3)
    mz = value[:, 0].astype(float)
    intensity = value[:, 1].astype(float)
    spectrum = Spectrum(mz, intensity, mz_precision=MZ_PRECISION, metadata=spot_name)
    # TODO: Regex here is not universal, need adjustment from case to case
    x = re.findall(r'R00X(.*?)Y', spot_name)[0]
    x = int(x)
    y = re.findall(r'Y(.*?)$', spot_name)[0]
    y = int(y)
    return (x, y), spectrum


def msi_from_txt(txt_file_path: str) -> dict:
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
    with open(txt_file_path) as f:
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

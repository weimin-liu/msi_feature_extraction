import numpy as np


def get_tic_map(msi: dict):
    """
    this function take in the msi dict, and return the image matrix of tic

    Parameters
    -------

    msi: a dictionary-like object

    Returns
    --------

    im: a matrix that has the same shape with the slide, i.e., the length of rows and columns of the matrix are the
        same as the width of the slide. The filled value is the corresponding TIC of the spectrum, 'nan' means no spectrum was generated at that spot.

    """
    tics = dict()
    for key in msi:
        tics[key] = np.sum(msi[key].intensity_values)

    xy = np.array(list(tics.keys()))
    tic = np.array(list(tics.values()))
    tics = np.hstack((xy, tic.reshape(-1, 1)))

    xLocation = tics[:, 0].astype(int)
    yLocation = tics[:, 1].astype(int)
    xLocation = xLocation - min(xLocation)
    yLocation = yLocation - min(yLocation)
    col = max(np.unique(xLocation))
    row = max(np.unique(yLocation))
    im = np.zeros((col, row))
    im[:] = np.nan

    for i in range(len(xLocation)):
        im[np.asscalar(xLocation[i]) - 1, np.asscalar(yLocation[i] - 1)] = tic[i]

    return im

from sklearn.cluster import KMeans
import matplotlib as mpl

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from sklearn.preprocessing import MaxAbsScaler


def grid_data(x, y, z, interp=False):
    if interp:
        points = np.array([x, y]).T
        values = z
        grid_x, grid_y = np.mgrid[x.min():x.max():1, y.min():y.max():1]
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    else:
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        z = np.array(z).flatten()
        df = pd.DataFrame(np.stack([x, y, z]).T, columns=['x', 'y', 'z'])
        grid_z = pd.pivot_table(df, values='z', index='x', columns='y')
    return grid_z


def imshow(x, y, z, interp=False, title=None, *args, **kwargs):
    grid_z0 = grid_data(x, y, z, interp=interp)
    fig, ax = plt.subplots()
    im = ax.imshow(grid_z0, *args, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.yaxis.get_offset_text().set_visible(False)
    fig.tight_layout()
    return ax


def show_kmeans(spot, features, n_clusters=3, *args, **kwargs):
    kmeans = KMeans(n_clusters=n_clusters)
    features = MaxAbsScaler().fit_transform(features)
    kmeans.fit(features)
    cmap = mpl.cm.viridis
    bounds = [i - 0.5 for i in range(n_clusters + 1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    grid_z0 = grid_data(spot[:, 0], spot[:, 1], kmeans.labels_)
    fig, ax = plt.subplots()
    im = ax.imshow(grid_z0, cmap=cmap, norm=norm, *args, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax)
    plt.show()
    return kmeans


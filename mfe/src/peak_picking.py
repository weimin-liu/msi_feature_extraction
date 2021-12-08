import pandas as pd
import numpy as np
import tqdm
from skimage.feature.texture import greycomatrix
from scipy import interpolate
from skimage import exposure

# === The matrix used to calculate texture score === #
C8 = np.zeros((8, 8))
C8[0, 0] = 4
C8[0, 1] = 2
C8[1, 0] = 2
C8[0, 2] = 1
C8[2, 0] = 1
C8[1, 1] = 1
C8[-1, -1] = 4
C8[-1, -2] = 2
C8[-2, -1] = 2
C8[-1, -3] = 1
C8[-3, -1] = 1
C8[-2, -2] = 1

C4 = np.zeros((4, 4))
C4[0, 0] = 2
C4[0, 1] = 1
C4[1, 0] = 1
C4[-1, -1] = 2
C4[-1, -2] = 1
C4[-2, -1] = 1
# =================================================== #


def interpolate_missing_pixels(
        image: np.ndarray,
        fill_value: int = 0,
        method: str = 'nearest'
):
    """
    Parameters:

        image: a 2D image, from which the edge whitespace have already been removed.

        fill_value: the value with which to fill the missing pixels when interpolation is not possible

        method: interpolation method, one of 'nearest', 'linear', 'cubic'.

    Return:

        the image with missing values interpolated
    """

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    image = np.ma.masked_invalid(image)

    known_x = xx[~image.mask]
    known_y = yy[~image.mask]
    known_v = image[~image.mask]
    missing_x = xx[image.mask]
    missing_y = yy[image.mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def de_flatten(coordinates: np.ndarray, peaks: np.ndarray):
    """
    Parameters:

        coordinates: array with x, y coordinates

        peaks: 1d array of intensity

    Return:

        de-flattened array that has the shape of same as the slide
    """
    x_location = coordinates[:, 0].astype(int)
    y_location = coordinates[:, 1].astype(int)
    min_x = min(x_location)
    min_y = min(y_location)
    x_location = x_location - min_x
    y_location = y_location - min_y
    col = max(np.unique(x_location))
    row = max(np.unique(y_location))
    im = np.zeros((col + 1, row + 1))
    im[:] = np.nan
    for i in range(len(x_location)):
        im[x_location[i], y_location[i]] = peaks[i]
    return im


def remove_whitespace(image: np.ndarray):
    """
    The original image of the slide is slightly crooked, therefore, a slight rotation followed by removing whitespace
    is needed before interpolating missing pixels. Note that the rotation angel and whitespace threshold is highly
    specific from case to case, do not use this function without modification!

    Parameters:
        image: the 2d array of image that is crooked

    Returns:

        image array that is correctly rotated with whitespace removed
    """

    # image = rotate(image, angle=-2, mode='wrap')
    #
    # whitespace_col = 1 - np.count_nonzero(np.isnan(image), axis=0) / len(image)
    # image = image[:, 4:40]
    #
    # whitespace_row = 1 - np.count_nonzero(np.isnan(image), axis=1) / len(image.T)
    # image = image[4:249]

    # ideal for the 0-5cm slice, only cut out the whitespace without rotation, to avoid any false interpolation
    image = image[26:235, 3:-5]

    return image


def contrast_stretching(image: np.ndarray):
    """
    The ion image may have hotspot, thus influence the following analysis. In contrast stretching, the image is
    rescaled to include all intensities that fall within the 2nd and 98th percentiles. For comparing with histogram
    equalization, seeing https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html

    Parameters:

        image: the 2d array of image that may have hotspot

    Returns:

        image array with hotspot removed

    """
    p2, p98 = np.nanpercentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    return image


def cal_structure_score(image, q=4):
    """
    Quantize the image and then calculate its structure score using grey-level co-occurrence matrix. See
    https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html

    Parameters:

        image: the 2d image array

        q: in order to get a more fair structure score, for each ion image, their intensities are evenly divided into
        bins with a number of q, and the intensities are then replaced with the label (integer number). Currently,
        q is fixed to 4 for the convenience of the following structure score calculation.

    Returns:

        The structure score of the image. The higher the score, the more structured the ion image.

    """
    nan_flg = False
    df = pd.DataFrame(image.reshape(-1, 1))
    n_spots = np.count_nonzero(df[0].to_numpy())
    q = q
    bin_labels = list(range(q))
    try:
        if df[0].min() == 0:
            df = df.replace(0, np.nan)
            nan_flg = True
        df['quantile_ex_1'] = pd.qcut(df[0], q=q, labels=bin_labels, duplicates='drop')
        im_quantized = df['quantile_ex_1'].to_numpy().reshape(image.shape)
        im_quantized[np.isnan(im_quantized)] = q
        im_quantized = im_quantized.astype(int)
        if nan_flg:
            gcm = greycomatrix(im_quantized, [1], [3 * np.pi / 4], levels=q+1)[:, :, 0, 0]
            gcm = gcm[0:-1, 0:-1]
        else:
            gcm = greycomatrix(im_quantized, [1], [3 * np.pi / 4], levels=q)[:, :, 0, 0]
        score = np.sum(np.multiply(gcm, C4)) / n_spots

    except ValueError:
        score = 0

    return score


def get_peak_ranks(feature_table: pd.DataFrame):
    """
    This is an example of how to get structued ion image from feature table.

    Parameters:

        feature_table: a DataFrame object

    Returns:

        t_df: structure scores for each ion image

        deflated_arr: a 3d array with ion image ravelled

    """

    dirty_df = feature_table

    dirty_df = dirty_df.iloc[:, 1:]

    spot = dirty_df[['x', 'y']].to_numpy()

    arr = dirty_df.drop(columns=['x', 'y']).to_numpy()

    mzs = list(dirty_df.drop(columns=['x', 'y']).columns)
    mzs = np.array([float(mz) for mz in mzs])

    t = dict()

    deflated_arr = list()

    for mz in tqdm.tqdm(list(mzs)):
        test = arr[:, mzs == mz]
        image = de_flatten(spot, test)

        image = contrast_stretching(image)

        image = remove_whitespace(image)

        # image = interpolate_missing_pixels(image)

        q = 4

        score = cal_structure_score(image, q)

        deflated_arr.append(image)

        t[mz] = score

    t_df = pd.DataFrame.from_dict(t, orient='index')

    return t_df, deflated_arr


def sel_peak_by_rank(t_df, deflated_arr, feature_table: pd.DataFrame, threshold):
    """
    Select the structured peaks (thus more meaningful) in the feature table and return the result

    Parameters:
    --------
        t_df: a DataFrame object with ranked mass-to-charge ratios

        feature_table: a Dataframe object

        threshold: above which the peaks will be preserved

    Returns:
    --------
        feature_table_picked: a Dataframe object with only peaks that have above threshold structure score

        deflated_arr_picked: an array with picked ion images
    """

    t_df_arr = t_df.to_numpy()

    mask = t_df_arr >= threshold

    deflated_arr_picked = deflated_arr[mask]

    deflated_arr_picked = np.array(deflated_arr_picked)

    t_df = t_df[t_df >= threshold]

    sel_mzs = list(t_df.index)

    sel_mzs.extend(['x', 'y'])

    sel_columns = [col for col in feature_table.columns() if col in sel_mzs]

    feature_table_picked = feature_table[sel_columns]

    return feature_table_picked, deflated_arr_picked





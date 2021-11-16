import pandas as pd
import numpy as np
import tqdm
from skimage.feature.texture import greycomatrix
from scipy import interpolate
from skimage.transform import rotate
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
    x_location = x_location - min(x_location)
    y_location = y_location - min(y_location)
    col = max(np.unique(x_location))
    row = max(np.unique(y_location))
    im = np.zeros((col, row))
    im[:] = np.nan
    for i in range(len(x_location)):
        im[np.asscalar(x_location[i]) - 1, np.asscalar(y_location[i] - 1)] = peaks[i]
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

    image = rotate(image, angle=-2, mode='wrap')

    whitespace_col = 1 - np.count_nonzero(np.isnan(image), axis=0) / len(image)
    image = image[:, 4:40]

    whitespace_row = 1 - np.count_nonzero(np.isnan(image), axis=1) / len(image.T)
    image = image[4:249]

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
        # TODO: ion image that cannot be quantized into 4 parts will be threw away, further examination may be required.

    Returns:

        The structure score of the image. The higher the score, the more structured the ion image.

    """
    df = pd.DataFrame(image.reshape(-1, 1))
    q = q
    bin_labels = list(range(q))
    try:
        df['quantile_ex_1'] = pd.qcut(df[0], q=q, labels=bin_labels, duplicates='drop')
        im_quantized = df['quantile_ex_1'].to_numpy().reshape(image.shape)
        gcm = greycomatrix(im_quantized, [1], [3 * np.pi / 4], levels=q)[:, :, 0, 0]
        score = np.sum(np.multiply(gcm, C4))

    except ValueError:
        score = 0
        # df['quantile_ex_1'] = pd.qcut(df[0], q=q, duplicates='drop')
        # q = len(df['quantile_ex_1'].unique())
        # bin_labels = list(range(q))
        # df['quantile_ex_1'] = pd.qcut(df[0], q=q, labels=bin_labels, duplicates='drop')

    return score


def main(feature_table_path=None, score_threshold=6000):
    """
    This is an example of how to get structued ion image from feature table.

    Parameters:

        feature_table_path: path to the feature table

        score_threshold: the threshold for preserving structured ra

    Returns:

        t_df: structure scores for each ion image

        deflated_arr: a 3d array with ion image ravelled

        selected_mz: the list of mzs that have structured images.
    """
    if feature_table_path is None:
        feature_table_path = "../../../examples/SBB5-10cm_mz520-580.csv"
    else:
        feature_table_path = feature_table_path

    dirty_df = pd.read_csv(feature_table_path)
    dirty_df = dirty_df.iloc[:, 1:]

    spot = dirty_df[['x', 'y']].to_numpy()

    arr = dirty_df.drop(columns=['x', 'y']).to_numpy()

    mzs = list(dirty_df.drop(columns=['x', 'y']).columns)
    mzs = np.array([float(mz) for mz in mzs])

    t = dict()
    selected_mz = list()
    deflated_arr = list()

    for mz in tqdm.tqdm(list(mzs)):
        test = arr[:, mzs == mz]
        image = de_flatten(spot, test)
        image = remove_whitespace(image)
        image = interpolate_missing_pixels(image)
        image = contrast_stretching(image)
        q = 4
        score = cal_structure_score(image, q)
        if score >= score_threshold:
            selected_mz.append(mz)
            deflated_arr.append(image)
            t[mz] = score
    t_df = pd.DataFrame.from_dict(t, orient='index')
    return t_df, deflated_arr, selected_mz


if __name__ == "__main__":
    structure_score, im, mzs = main()
    im_f = [arr.flatten() for arr in im]
    im_f = np.array(im_f)

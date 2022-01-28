import numpy as np
import pandas as pd
import tqdm
from scipy import interpolate
from skimage.feature import graycomatrix
from scipy.interpolate import interpolate
from skimage.exposure import exposure
import cv2


class GLCMPeakRanking:
    # === The matrix used to calculate texture score === #
    C = np.array([
        [4, 2, 1, 0, 0],
        [2, 1, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 1, 2],
        [0, 0, 1, 2, 4]
    ])

    def __init__(self,
                 fill_value=0,
                 interpolation='blur',
                 blur_filter=(2, 2),
                 remove_whitespace=False,
                 stretch_contrast=True,
                 contrast_lim=(2, 98),
                 q=5):
        """
        Args:
            fill_value: the value with which to fill the missing pixels when interpolation is not possible
            interpolation: interpolation method, one of 'nearest', 'linear', 'cubic'.
            remove_whitespace:
            stretch_contrast:
            contrast_lim:
            q: in order to get a more fair structure score, for each ion image, their intensities are evenly divided into
            bins with a number of q, and the intensities are then replaced with the label (integer number). Currently,
            q is fixed to 5 for the convenience of the following structure score calculation.
        """
        if self.C.shape[0] == self.C.shape[0] == q:
            self.interpolation = interpolation
            self.fill_value = fill_value
            self.remove_whitespace = remove_whitespace
            self.stretch_contrast = stretch_contrast
            self.contrast_lim = contrast_lim
            self.blur_filter = blur_filter
            self.q = q
            self.feature_table = pd.DataFrame()
            self.images = list()
            self.score = pd.DataFrame()
            self.mzs = np.array([])
        else:
            raise ValueError('The number of zones are not equal to the calculation matrix!')

    def _reset(self):
        if len(self.score) != 0:
            del self.feature_table
            del self.images
            del self.score

    def fit(self, feature_table: pd.DataFrame, angle=0):
        """
        Parameters:
        --------
        """
        self._reset()
        return self.partial_fit(feature_table, angle=angle)

    def partial_fit(self, feature_table: pd.DataFrame, angle=0):
        """
        This is an example of how to get structured ion image from feature table.

        Parameters:

            angle:
            feature_table: a DataFrame object

        Returns:

            t_df: structure scores for each ion image

            deflated_arr: a 3d array with ion image ravelled

        """
        self.feature_table = feature_table

        dirty_df = self.feature_table

        spot = dirty_df[['x', 'y']].to_numpy()

        arr = dirty_df.drop(columns=['x', 'y']).to_numpy()

        mzs = list(dirty_df.drop(columns=['x', 'y']).columns)
        self.mzs = np.array([float(mz) for mz in mzs])

        t = dict()

        print('Processing each ion image')

        for mz in tqdm.tqdm(list(self.mzs)):
            test = arr[:, self.mzs == mz]

            image = de_flatten(spot, test,
                               remove_whitespace=self.remove_whitespace,
                               stretch_contrast=self.stretch_contrast,
                               contrast_lim=self.contrast_lim,
                               interpolation=self.interpolation,
                               blur_filter=self.blur_filter,
                               fill_value=self.fill_value)

            score = self.cal_structure_score(image, angle)

            self.images.append(image)

            t[mz] = score

        self.score = pd.DataFrame.from_dict(t, orient='index')

        self.images = np.array(self.images)

        self.images[np.isnan(self.images)] = 0

    @property
    def mzs_sorted(self):
        """
        Return mz values of the ion images sorted by ranking structured score

        Returns:
        """
        score_sorted = self.score.sort_values(by=0, ascending=False)
        mzs_sorted = np.array(list(score_sorted.index))
        return mzs_sorted

    @property
    def images_sorted(self):
        """
        return image arrays that are sorted by ranking structured score

        Returns:

        """
        score = self.score.to_numpy().flatten()
        score_increase = score.argsort()
        images_sorted = self.images[score_increase[::-1]]
        return images_sorted

    def cal_structure_score(self, image, angle):
        """
        Quantize the image and then calculate its structure score using grey-level co-occurrence matrix. See
        https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html

        Parameters:

            angle:
            image: the 2d image array

        Returns:

            The structure score of the image. The higher the score, the more structured the ion image.

        """
        nan_flg = False

        df = pd.DataFrame(image.reshape(-1, 1))

        n_spots = np.count_nonzero(df[0].to_numpy())

        bin_labels = list(range(self.q))

        if n_spots == 0:

            score = 0

        else:

            try:

                if df[0].min() == 0:
                    df = df.replace(0, np.nan)

                    nan_flg = True

                df['quantile_ex_1'] = pd.qcut(df[0], q=self.q, labels=bin_labels, duplicates='drop')

                im_quantized = df['quantile_ex_1'].to_numpy().reshape(image.shape)

                im_quantized[np.isnan(im_quantized)] = self.q

                im_quantized = im_quantized.astype(int)

                if nan_flg:

                    gcm = graycomatrix(im_quantized, [1], [angle], levels=self.q + 1)[:, :, 0, 0]

                    gcm = gcm[0:-1, 0:-1]

                else:

                    gcm = graycomatrix(im_quantized, [1], [angle], levels=self.q)[:, :, 0, 0]

                score = np.sum(np.multiply(gcm, self.C)) / n_spots

            except ValueError:

                score = 0

        return score

    def transform(self, threshold):
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

        t_df_arr = self.score.to_numpy().flatten()

        mask = t_df_arr >= threshold

        images_picked = self.images[mask]

        images_picked = np.array(images_picked)

        t_df = self.score[self.score >= threshold]

        t_df = t_df.dropna()

        sel_mzs = list(t_df.index)

        sel_mzs.extend(['x', 'y'])

        sel_columns = [col for col in list(self.feature_table.columns) if col in sel_mzs]

        feature_table_picked = self.feature_table[sel_columns]

        return feature_table_picked, images_picked


def whitespace_removal(image: np.ndarray):
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
    # image = image[26:235, 3:-5]

    return image


def contrast_stretching(image: np.ndarray,
                        contrast_lim=(2, 98)):
    """
    The ion image may have hotspot, thus influence the following analysis. In contrast stretching, the image is
    rescaled to include all intensities that fall within the 2nd and 98th percentiles. For comparing with histogram
    equalization, seeing https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html

    Parameters:

        contrast_lim:
        image: the 2d array of image that may have hotspot

    Returns:

        image array with hotspot removed

    """
    p_low, p_high = np.nanpercentile(image, contrast_lim)
    image = exposure.rescale_intensity(image, in_range=(p_low, p_high))
    return image


def de_flatten(coordinates: np.ndarray,
               peaks: np.ndarray,
               remove_whitespace=False,
               stretch_contrast=True,
               contrast_lim=(2, 98),
               interpolation='blur',
               blur_filter=(2, 2),
               fill_value=0
               ):
    """
    Parameters:

        fill_value:
        blur_filter:
        interpolation:
        contrast_lim:
        remove_whitespace:
        stretch_contrast:
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
    im[:] = 0
    for i in range(len(x_location)):
        im[x_location[i], y_location[i]] = peaks[i]
    if stretch_contrast:
        im = contrast_stretching(im, contrast_lim)
    if remove_whitespace:
        im = whitespace_removal(im)
    if interpolation != 'None':
        im = image_interpolation(im, method=interpolation, blur_filter=blur_filter, fill_value=fill_value)
    return im


def image_interpolation(
        image: np.ndarray,
        method='blur',
        blur_filter=(2, 2),
        fill_value=0
):
    """
    Parameters:

        fill_value:
        method:
        blur_filter:
        image: a 2D image, from which the edge whitespace have already been removed.

    Return:

        the image with missing values interpolated
    """

    if method == 'blur':

        image[np.isnan(image)] = fill_value

        kernel = np.ones((blur_filter[0], (blur_filter[1])), np.float32) / (
                blur_filter[0] * blur_filter[1])

        interp_image = cv2.filter2D(image, -1, kernel)

    else:
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

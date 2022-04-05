import itertools

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import interpolate
from skimage.feature import graycomatrix, graycoprops
from scipy.interpolate import interpolate
from skimage.exposure import exposure
from cv2 import cv2
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class GLCMPeakRanking:
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
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.remove_whitespace = remove_whitespace
        self.stretch_contrast = stretch_contrast
        self.contrast_lim = contrast_lim
        self.blur_filter = blur_filter
        self.q = q
        self.feature_table = pd.DataFrame()
        self.images = list()
        self.results = list()
        self.prop = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        self.mzs = np.array([])

    def fit(self, feature_table: pd.DataFrame, dist, angle):
        """
        Parameters:
        --------
        """
        return self.partial_fit(feature_table, dist, angle)

    def partial_fit(self, feature_table: pd.DataFrame, dist, angle):
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

            self.cal_glcm_prop(image, dist, angle)

            self.images.append(image)

        self.images = np.array(self.images)

        self.images[np.isnan(self.images)] = 0

    def cal_glcm_prop(self, image, dist, angle):
        """
        Quantize the image and then calculate its structure score using grey-level co-occurrence matrix. See
        https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html

        Parameters:

            angle:
            image: the 2d image array

        Returns:

            The structure score of the image. The higher the score, the more structured the ion image.

        """

        df = pd.DataFrame(image.reshape(-1, 1))

        n_spots = np.count_nonzero(df[0].to_numpy())

        bin_labels = list(range(1, self.q + 1))

        result = list()

        if n_spots == 0:

            result.append(np.nan)

        else:

            try:

                if df[0].min() == 0:
                    df = df.replace(0, np.nan)

                df['quantile_ex_1'] = pd.qcut(df[0], q=self.q, labels=bin_labels, duplicates='drop')

                im_quantized = df['quantile_ex_1'].to_numpy().reshape(image.shape)

                im_quantized[np.isnan(im_quantized)] = 0

                im_quantized = im_quantized.astype(int)

                gcm = graycomatrix(im_quantized, dist, angle, levels=self.q + 1)

                for d, theta in itertools.product(range(len(dist)), range(len(angle))):

                    for key in self.prop:
                        result.append(graycoprops(gcm, key)[d, theta])

            except ValueError:
                result.append(np.nan)

        self.results.append(result)

    def fancy_overview(self, save_path, xlim, ylim):
        results = pd.DataFrame(self.results)
        results = results.dropna()
        results.index = self.mzs[list(results.index)]
        similarities = pd.DataFrame(cosine_similarity(results), index=results.index, columns=list(results.index))
        pca = PCA(n_components=2)
        X = StandardScaler().fit_transform(results)
        pc = pca.fit_transform(X)
        pc = pd.DataFrame(pc, index=results.index)
        coef = pd.DataFrame(pca.components_.T)
        coef['label'] = coef.index.map(lambda x: x % len(self.prop))
        color = ['g', 'k', 'm', 'y', 'c']
        fig = plt.figure()
        ax = fig.add_subplot(111, label='1')
        ax2 = fig.add_subplot(111, label='2', frame_on=False)
        xs = pc.iloc[:, 0]
        ys = pc.iloc[:, 1]
        n = len(coef)
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        sc = ax.scatter(xs * scalex, ys * scaley, alpha=0.5, c=similarities[0], vmin=similarities[0].quantile(0.15), vmax=similarities[0].quantile(0.85),cmap='jet')

        pc1_str = round(100 * pca.explained_variance_ratio_[0], 2)
        pc2_str = round(100 * pca.explained_variance_ratio_[1], 2)
        ax.set_xlabel(f'PC1 ({pc1_str:.2f}%)')
        ax.set_ylabel(f'PC2 ({pc2_str:.2f}%)')
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        for i in range(n):
            ax2.arrow(0, 0, coef.iloc[i, 0], coef.iloc[i, 1], color=color[coef.loc[i, 'label']], alpha=0.5,
                      linewidth=0.1)
        ax2.xaxis.tick_top()
        ax2.yaxis.tick_right()
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_xticks([xlim[0], 0, xlim[1]])
        ax2.set_yticks([ylim[0], 0, ylim[1]])

        ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')

        legend_elements = list()

        for i in range(len(color)):
            legend_elements.append(Line2D([0], [0], color=color[i], lw=2, label=f'$\it{self.prop[i][0:4]}$'))
        ax.legend(handles=legend_elements, loc='lower left',  fancybox=True, shadow=True,ncol=3)
        plt.grid()
        cbar_ax = fig.add_axes([0.15, 0.8, 0.2, 0.03])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
        cbar.ax.set_xticks([similarities[0].quantile(0.15), similarities[0].quantile(0.85)])
        cbar.ax.set_yticks([])
        cbar.ax.set_xticklabels(['far', 'near'])
        plt.savefig(save_path, format='svg')
        plt.show()

    @property
    def distance(self) -> pd.DataFrame:
        results = pd.DataFrame(self.results)
        results = results.dropna()
        results.index = self.mzs[list(results.index)]
        similarities = pd.DataFrame(cosine_similarity(results), index=results.index, columns=list(results.index))
        distance = similarities[0].iloc[:-1]
        distance = distance.sort_values(ascending=False)
        return distance

    def mz_at_percentile(self, percentile: list) -> list:
        mzs = [
            list(self.distance[self.distance >= self.distance.sort_values(ascending=False).quantile(i / 100)].index)[-1]
            for i in percentile]
        return mzs

    def mzs_above_percentile(self, percentile: int) -> list:
        mzs = list(
            self.distance[self.distance >= self.distance.sort_values(ascending=False).quantile(percentile / 100)].index)
        return mzs

    # def transform(self, threshold):
    #     """
    #     Select the structured peaks (thus more meaningful) in the feature table and return the result
    #
    #     Parameters:
    #     --------
    #         t_df: a DataFrame object with ranked mass-to-charge ratios
    #
    #         feature_table: a Dataframe object
    #
    #         threshold: above which the peaks will be preserved
    #
    #     Returns:
    #     --------
    #         feature_table_picked: a Dataframe object with only peaks that have above threshold structure score
    #
    #         deflated_arr_picked: an array with picked ion images
    #     """
    #
    #     t_df_arr = self.score.to_numpy().flatten()
    #
    #     mask = t_df_arr >= threshold
    #
    #     images_picked = self.images[mask]
    #
    #     images_picked = np.array(images_picked)
    #
    #     t_df = self.score[self.score >= threshold]
    #
    #     t_df = t_df.dropna()
    #
    #     sel_mzs = list(t_df.index)
    #
    #     sel_mzs.extend(['x', 'y'])
    #
    #     sel_columns = [col for col in list(self.feature_table.columns) if col in sel_mzs]
    #
    #     feature_table_picked = self.feature_table[sel_columns]
    #
    #     return feature_table_picked, images_picked


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
        image,
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

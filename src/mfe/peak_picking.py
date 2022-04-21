"""
this module contains functions for peak picking and peak fitting
"""
import itertools

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from skimage.feature import graycomatrix, graycoprops
from skimage.exposure import exposure
import cv2

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class GLCMPeakRanking:
    """
    class for peak ranking based on GLCM properties of a given image

    """

    PROP = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    CONTRAST_LIM = (2, 98)

    def __init__(self,
                 interpolation='blur',
                 blur_filter=(2, 2),
                 q_val=5):
        """
        Args: fill_value: the value with which to fill the missing pixels when interpolation is
        not possible interpolation: interpolation method, one of 'nearest', 'linear', 'cubic'.
        contrast_lim: q: in order to get a more fair
        structure score, for each ion image, their intensities are evenly divided into bins with
        a number of q, and the intensities are then replaced with the label (integer number).
        Currently, q is fixed to 5 for the convenience of the following structure score
        calculation.
        """
        self.interpolation = interpolation
        self.blur_filter = blur_filter
        self.quantile = q_val
        self.feature_table = pd.DataFrame()
        self.images = []
        self.results = []
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

        for mz_val in tqdm.tqdm(list(self.mzs)):
            test = arr[:, self.mzs == mz_val]

            image = de_flatten(spot, test,
                               contrast_lim=self.CONTRAST_LIM,
                               interpolation=self.interpolation,
                               blur_filter=self.blur_filter)

            self.cal_glcm_prop(image, dist, angle)

            self.images.append(image)

        self.images = np.array(self.images)

        self.images[np.isnan(self.images)] = 0

    def cal_glcm_prop(self, image, dist, angle):
        """
        Quantize the image and then calculate its structure score using grey-level co-occurrence
        matrix. See https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html

        Parameters:

            angle:
            image: the 2d image array

        Returns:

            The structure score of the image. The higher the score, the more structured the ion
            image.

        """

        df_new = pd.DataFrame(image.reshape(-1, 1))

        n_spots = np.count_nonzero(df_new[0].to_numpy())

        bin_labels = list(range(1, self.quantile + 1))

        result = []

        if n_spots == 0:

            result.append(np.nan)

        else:

            try:

                if df_new[0].min() == 0:
                    df_new = df_new.replace(0, np.nan)

                df_new['quantile_ex_1'] = pd.qcut(df_new[0], q=self.quantile, labels=bin_labels,
                                                  duplicates='drop')

                im_quantized = df_new['quantile_ex_1'].to_numpy().reshape(image.shape)

                im_quantized[np.isnan(im_quantized)] = 0

                im_quantized = im_quantized.astype(int)

                gcm = graycomatrix(im_quantized, dist, angle, levels=self.quantile + 1)

                for d_val, theta in itertools.product(range(len(dist)), range(len(angle))):

                    for key in self.PROP:
                        result.append(graycoprops(gcm, key)[d_val, theta])

            except ValueError:
                result.append(np.nan)

        self.results.append(result)

    # pylint: disable=too-many-locals
    def fancy_overview(self, save_path, x_lim, y_lim):
        """
        This is an example of how to get a fancy overview of the variability in ion images.
        Args:
            save_path:
            x_lim:
            y_lim:

        Returns:

        """
        results = pd.DataFrame(self.results)
        results = results.dropna()
        results.index = self.mzs[list(results.index)]
        similarities = pd.DataFrame(cosine_similarity(results), index=results.index,
                                    columns=list(results.index))
        pca = PCA(n_components=2)
        results_std = StandardScaler().fit_transform(results)
        pcs = pca.fit_transform(results_std)
        pcs = pd.DataFrame(pcs, index=results.index)
        coef = pd.DataFrame(pca.components_.T)
        coef['label'] = coef.index.map(lambda x: x % len(self.PROP))
        color = ['g', 'k', 'm', 'y', 'c']
        fig = plt.figure()
        axs = fig.add_subplot(111, label='1')
        ax2 = fig.add_subplot(111, label='2', frame_on=False)
        xs_val = pcs.iloc[:, 0]
        ys_val = pcs.iloc[:, 1]
        len_coef = len(coef)
        scale_x = 1.0 / (xs_val.max() - xs_val.min())
        scale_y = 1.0 / (ys_val.max() - ys_val.min())
        sc_plot = axs.scatter(xs_val * scale_x, ys_val * scale_y, alpha=0.5, c=similarities[0],
                              vmin=similarities[0].quantile(0.15),
                              vmax=similarities[0].quantile(0.85), cmap='jet')

        pc1_str = round(100 * pca.explained_variance_ratio_[0], 2)
        pc2_str = round(100 * pca.explained_variance_ratio_[1], 2)
        axs.set_xlabel(f'PC1 ({pc1_str:.2f}%)')
        axs.set_ylabel(f'PC2 ({pc2_str:.2f}%)')
        axs.set_xticks([-1, 0, 1])
        axs.set_yticks([-1, 0, 1])
        axs.set_xlim(-1, 1)
        axs.set_ylim(-1, 1)
        for i in range(len_coef):
            ax2.arrow(0, 0, coef.iloc[i, 0], coef.iloc[i, 1], color=color[coef.loc[i, 'label']],
                      alpha=0.5, linewidth=0.1)
        ax2.xaxis.tick_top()
        ax2.yaxis.tick_right()
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)
        ax2.set_xticks([x_lim[0], 0, x_lim[1]])
        ax2.set_yticks([y_lim[0], 0, y_lim[1]])

        ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')

        legend_elements = []

        for i, _ in enumerate(color):
            legend_elements.append(Line2D([0], [0], color=color[i], lw=2,
                                          label=fr'$\it{self.PROP[i][0:4]}$'))
        axs.legend(handles=legend_elements, loc='lower left', fancybox=True, shadow=True, ncol=3)
        plt.grid()
        cbar_ax = fig.add_axes([0.15, 0.8, 0.2, 0.03])
        cbar = fig.colorbar(sc_plot, cax=cbar_ax, orientation='horizontal')
        cbar.ax.set_xticks([similarities[0].quantile(0.15), similarities[0].quantile(0.85)])
        cbar.ax.set_yticks([])
        cbar.ax.set_xticklabels(['far', 'near'])
        plt.savefig(save_path, format='svg')
        plt.show()

    @property
    def distance(self) -> pd.DataFrame:
        """
        measure the distance between the features
        Returns:

        """
        results = pd.DataFrame(self.results)
        results = results.dropna()
        results.index = self.mzs[list(results.index)]
        similarities = pd.DataFrame(cosine_similarity(results), index=results.index,
                                    columns=list(results.index))
        distance = similarities[0].iloc[:-1]
        distance = distance.sort_values(ascending=False)
        return distance

    def mz_at_percentile(self, percentile: list) -> list:
        """
        get the mz at a certain percentile
        Args:
            percentile:

        Returns:

        """
        mzs = [
            list(self.distance[
                     self.distance >=
                     self.distance.sort_values(ascending=False).quantile(i / 100)].index)[-1]
            for i in percentile]
        return mzs

    def mzs_above_percentile(self, percentile: int) -> list:
        """
            get the mzs above a certain percentile       Args:
            percentile:

        Returns:

        """
        mzs = list(
            self.distance[self.distance >= self.distance.sort_values(ascending=False).
                quantile(percentile / 100)].index)
        return mzs

    # def transform(self, threshold): """ Select the structured peaks (thus more meaningful) in
    # the feature table and return the result
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
    #         feature_table_picked: a Dataframe object with only peaks that have above
    #         threshold structure score
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


def contrast_stretching(image: np.ndarray,
                        contrast_lim=(2, 98)):
    """
    The ion image may have hotspot, thus influence the following analysis. In contrast
    stretching, the image is rescaled to include all intensities that fall within the 2nd and
    98th percentiles. For comparing with histogram equalization, seeing
    https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html

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
               contrast_lim=(2, 98),
               interpolation='blur',
               blur_filter=(2, 2)
               ):
    """
    Parameters:
        blur_filter:
        interpolation:
        contrast_lim:
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
    image = np.zeros((col + 1, row + 1))
    image[:] = 0
    for i, _ in enumerate(x_location):
        image[x_location[i], y_location[i]] = peaks[i]
    image = contrast_stretching(image, contrast_lim)
    if interpolation != 'None':
        image = image_interpolation(image, method=interpolation,
                                    blur_filter=blur_filter, fill_value=0)
    return image


def image_interpolation(
        image,
        method='blur',
        blur_filter=(2, 2),
        fill_value=0
):
    """
    Parameters:
        fill_value:
        image: 2d array
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
        # pylint: disable=no-member
        interp_image = cv2.filter2D(image, -1, kernel)

    else:
        return NotImplementedError
    return interp_image

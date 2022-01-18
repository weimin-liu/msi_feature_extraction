import numpy as np
import tqdm

from ..from_txt import get_ref_peaks, create_feature_table
from ..util import metrics


def search_peak_th(raw_data: dict, peak_th_candidates: list, peak_picking_method='prominent') -> dict:
    """
    This toolbox function can be used to decide which peak_th parameter should be used to get reference peaks. It
    will return four metrics for consideration:

    - Number of reference peaks: the number of reference peaks discovered

    - TIC Coverage: how many percentage of the peak intensity is covered after peak alignment

    - Accuracy: the overall m/z error between the measured peaks and the reference peak, including mean and std

    - Sparsity: how sparse is the feature table that is obtained from the reference peaks

    Args:
        raw_data: the raw data in dictionary form, returned by msi_from_txt()

        peak_th_candidates: a list of candidates to be considered

        peak_picking_method: the method for peak alignment, default to 'prominent'

    Returns:

    """

    # TODO: add sanity check before processing

    n_ref = list()

    cover = list()

    me = list()

    mstd = list()

    spar = list()

    for peak_th in tqdm.tqdm(peak_th_candidates):

        ref = get_ref_peaks(raw_data, peak_picking_method=peak_picking_method, peak_th=peak_th)

        n_ref.append(len(ref))

        feature_table, err_table = create_feature_table(raw_data, ref)

        coverage = metrics.peak_alignment_evaluation(raw_data, feature_table)

        cover.append(coverage['TIC_coverage'].mean())

        me.append(err_table.drop(columns=['x', 'y']).mean(skipna=True).mean(skipna=True))

        mstd.append(err_table.drop(columns=['x', 'y']).std(skipna=True).mean(skipna=True))

        spar.append((feature_table.drop(columns=['x', 'y']).to_numpy() == 0).mean())

    return {
        'n_ref': n_ref,
        'tic_coverage': cover,
        'mean_error': me,
        'mean_std': mstd,
        'sparsity': spar
    }







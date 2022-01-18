import numpy as np
import pandas as pd


def peak_alignment_evaluation(spectrum_dict: dict, feature_table: pd.DataFrame) -> dict:
    """
    Use this function to evaluate the performance of peak alignment. It has two metrics, the first is the coverage of
    the intensity of the aligned peaks, the second is the accuracy, which measures the discrepancy of the reference
    peaks comparing to the measured peaks before alignment.

    Args:
        spectrum_dict: the dictionary representing raw data

        feature_table: the feature table obtained after peak alignment

    Returns:

        coverage_dict: the percentage of tic that's been picked after alignment

    """

    if len(spectrum_dict) != len(feature_table):
        raise ValueError("This feature table is not derived from the current raw data!")
    else:

        feature = feature_table
        feature = feature.set_index(['x', 'y'])

        keys = list(spectrum_dict.keys())

        keys_df = pd.DataFrame(keys)

        keys_df = keys_df.set_index([0, 1])

        feature = feature.loc[keys_df.index, :]

        feature = np.array(feature)

        tic_after = np.sum(feature, axis=1)

        coverage_dict = {}

        for m in range(len(spectrum_dict)):

            key = list(spectrum_dict.keys())[m]

            coverage_dict[key] = tic_after[m] / spectrum_dict[key].tic

            if coverage_dict[key] > 1:

                raise ValueError('Something went wrong! TIC after alignment should not be greater than before!')

    return coverage_dict



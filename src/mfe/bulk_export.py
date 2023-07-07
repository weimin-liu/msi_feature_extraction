"""
this script is used for bulk exporting feature table from a directory of data analysis exports
"""
import os.path

from .from_txt import msi_from_txts, get_ref_peaks, create_feature_table


def get_txts(path: str):
    """
    get all txt files in a directory    :param path:  path
    :return: dictionary of measurement name and text file path
    """
    import os
    # get all the txt files in the directory/subdirectories, and group them to a list if they are in the same directory
    txts = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                if root in txts:
                    txts[root].append(file)
                else:
                    txts[root] = [file]
    return txts


def bulk_run(txts: dict, params: dict):
    """
    bulk run the mfe pipeline on the txt files
    :param txts: dictionary of measurement name and text file path
    :return: dictionary of measurement name and feature table
    """
    for measurement, txt in txts.items():
        if len(txt) > 1:
            txt = [os.path.join(measurement, f) for f in txt]
        else:
            txt = [os.path.join(measurement, txt[0])]
        feature_table, err_table = get_feature_table(txt, params)
        # save them to a csv file under  the same directory
        feature_table.to_csv(measurement + '_feature_table.csv')
        err_table.to_csv(measurement + '_err_table.csv')


def ask_for_params():
    """
    Ask for parameters for the mfe pipeline, including peak_th, normalization, etc.
    Returns:
        A dictionary containing the user-provided parameters.
    """
    params = {}

    # Ask for peak_th parameter
    peak_th = input("Enter the peak threshold value (sep multiple values by ,): ")
    if ',' in peak_th:
        peak_th = [float(x) for x in peak_th.split(',')]
    else:
        peak_th = [peak_th]
    params['peak_th'] = peak_th

    # Ask for normalization parameter
    normalization = input("Enter the normalization method (None or median): ")
    params['normalization'] = normalization

    # Ask for on parameter
    on = input("Enter the on parameter (e.g., R0, all): ")
    params['on'] = on

    return params


def get_feature_table(txt, params=None):
    """
    get feature table from txt file(s)
    Returns:

    """
    # get the mfe dict
    mfe_dict = msi_from_txts(txt)
    # find reference peaks
    ref_peaks = get_ref_peaks(mfe_dict, peak_th=params['peak_th'], on=params['on'])
    # find the peaks in the reference sample
    if len(params['peak_th']) > 1:
        print('Multiple peak_th values detected, will only run the pipeline on the first peak_th value')
    feature_table, error_table = create_feature_table(mfe_dict,
                                                      ref_peaks[params['peak_th'][0]],
                                                      normalization=params['normalization'])

    return feature_table, error_table


def main():
    # ask for the root directory
    root_dir = input("Enter the root directory: ")
    # get all the txt files in the directory/subdirectories, and group them to a list if they are in the same directory
    txts = get_txts(root_dir)
    # ask for parameters
    params = ask_for_params()
    # run the pipeline
    bulk_run(txts, params)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

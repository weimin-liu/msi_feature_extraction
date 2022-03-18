import pickle

import numpy as np
from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table
from argparse import ArgumentParser

def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="raw", help="input raw file(s), multiple files are seprated by comma", metavar="raw")
    parser.add_argument("-l", "--label", dest='label', help='label for the raw files, helpful if multiple raw files are provided', metavar='label')
    parser.add_argument("-o", "--output", dest="table", default='./feature_table.csv', help="output feature table", metavar="table")
    parser.add_argument("-pth", "--peak_th", dest="pth", default=0.1, help="peak prominence threshold for peak picking", metavar="pth")
    args = parser.parse_args()
    raw_path = args.raw.split(',')
    if len(raw_path) == 1:
        spectra = msi_from_txt(raw_path[0])
    else:
        spectra = dict()
        for path in raw_path:
            spectra[path] = msi_from_txt(path)


if __name__ == "__main__":


    for key in list(spectra1.keys()):
        spectra0[(-key[0], -key[1])] = spectra1[key]
    ref = get_ref_peaks(spectra0, peak_th=0.1)
    ref = ref[0.1]

    feature_table, error_table = create_feature_table(spectra0, ref, normalization='median')

    with open(ft, 'wb') as f:
        pickle.dump(feature_table, f)
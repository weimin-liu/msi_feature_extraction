import os
import pickle

import pandas as pd
from mfe.time import from_year_fraction


def get_all_instrumental_data(instrumental_data_dir):
    instrumental_data = dict()
    for root, dirs, files in os.walk(instrumental_data_dir):
        for file in files:
            if file.endswith(".csv"):
                instrumental_data[os.path.splitext(file)[0]] = pd.read_csv(os.path.join(root, file), delimiter='\t')
                instrumental_data[os.path.splitext(file)[0]].Age = instrumental_data[os.path.splitext(file)[0]].Age.map(
                    from_year_fraction)
                instrumental_data[os.path.splitext(file)[0]] = instrumental_data[os.path.splitext(file)[0]].set_index(
                    'Age')
    return instrumental_data


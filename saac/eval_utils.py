import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('once')


def load_occupation_data():
    pg_data_path = os.path.join('..', '..', 'data', 'prompt_generation')
    occupation_file = os.path.join(pg_data_path, 'interim', 'AnnualOccupations_TitleBank.csv')

    occupation_data = pd.read_csv(occupation_file)

    return occupation_data


def load_tda_data():
    pg_data_path = os.path.join('..', '..', 'data', 'prompt_generation')
    adjective_file = os.path.join(pg_data_path, 'interim', 'TDA_Bank.csv')

    cols = [0, 4, 5, 6]
    colnames = [
        'tda',
        'tda_compound',
        'tda_sentiment_cat',
        'tda_sentiment_val'
    ]
    tda_data = pd.read_csv(adjective_file, header=0, usecols=cols, names=colnames)

    return tda_data


# https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
def rgb_intensity(rgb_tuple):
    return np.mean(np.dot([0.2126, 0.7152, 0.0722], np.array(rgb_tuple)))


def rgb_sorter(rgb_tuples):
    # return sorted(rgb_tuples, key=lambda x: sum(x), reverse=False)
    return sorted(rgb_tuples, key=lambda x: rgb_intensity(x), reverse=False)



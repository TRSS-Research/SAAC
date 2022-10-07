import os
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


def rgb_sorter(rgb_tuples):
    return sorted(rgb_tuples, key=lambda x: sum(x), reverse=False)

import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

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


def generate_countplot(df, x_col, hue_col, title='', xlabel='', ylabel='', legend_title=''):
    """
    Generates a seaborn countplot using one column in the dataframe for x and one column for the hue.

    Args:
        dataframe (pandas.DataFrame): The dataframe to use for generating the countplot.
        x_col (str): The name of the column to use for the x-axis of the countplot.
        hue_col (str): The name of the column to use for the hue of the countplot.
        title: The title of the plot (default: empty string)
        xlabel: The label for the x-axis (default: empty string)
        ylabel: The label for the y-axis (default: empty string)
        legend_title: The title for the legend (default: empty string)

    """
    sns.set(style='darkgrid', palette='colorblind', color_codes=True)
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=x_col, hue=hue_col, data=df)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend = ax.legend(title=legend_title)
    legend.set_bbox_to_anchor((1, 1))
    plt.show()


def generate_histplot(df, x_col, hue_col, hue_order=None, kde=True, multiple='dodge', shrink=0.8, title='', xlabel='',
                      ylabel=''):
    sns.set(style='darkgrid', palette='colorblind', color_codes=True)
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=df, x=x_col, hue=hue_col, hue_order=hue_order, multiple=multiple, shrink=shrink, kde=kde)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def generate_displot(df, x_col, hue_col, kind="kde", title=None):
    sns.set(style='darkgrid', palette='colorblind', color_codes=True)
    plt.figure(figsize=(10, 6))
    sns.displot(data=df, x=x_col, hue=hue_col, kind=kind).set(title=title)


def rgb_histogram(df, x_col, rgb_col, n_bins=None, x_label=None, y_label=None, title=None):
    # Mostly just a visual test of intensity sorting per sentiment bin
    fig, ax1 = plt.subplots(1, 1)
    df = df.dropna(subset=[rgb_col])
    df_count, df_division = np.histogram(df[x_col], bins=n_bins)

    for idx in range(1, len(df_division)):
        if idx + 1 == len(df_division):
            mask = (df[x_col] >= df_division[idx - 1]) & (df[x_col] <= df_division[idx])
        else:
            mask = (df[x_col] >= df_division[idx - 1]) & (df[x_col] < df_division[idx])

        sorted_rgb = rgb_sorter(df[mask][rgb_col].apply(eval))

        for y, c in enumerate(sorted_rgb):
            plt.plot(df_division[idx - 1: idx + 1], y * np.ones(2), color=np.array(c) / 255)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)


def lumia_violinplot(df, x_col, rgb_col, n_bins=None, points_val=None, widths_val=None, y_label=None, x_label=None,
                     title=None):
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(1, 1)
    df = df.dropna(subset=[rgb_col])
    val_count, val_division = np.histogram(df[x_col], bins=n_bins)

    all_rgb_intensities = []

    for idx in range(1, len(val_division)):
        if idx + 1 == len(val_division):
            mask = (df[x_col] >= val_division[idx - 1]) & (df[x_col] <= val_division[idx])
        else:
            mask = (df[x_col] >= val_division[idx - 1]) & (df[x_col] < val_division[idx])

        if sum(mask) <= 0:
            continue

        rgb_intensities = df[mask][rgb_col].apply(eval).apply(rgb_intensity)
        all_rgb_intensities.append(list(rgb_intensities.values))

        parts = ax1.violinplot(rgb_intensities, positions=[np.mean(val_division[idx - 1:idx + 1])],
                               showmeans=True,
                               showextrema=False,
                               widths=widths_val,
                               points=points_val)

        hex_str = str(hex(int(np.median(rgb_intensities))))[2:]
        hex_color = f"#{hex_str}{hex_str}{hex_str}"

        for pc in parts['bodies']:
            pc.set_facecolor(hex_color)
            pc.set_edgecolor(hex_color)
            pc.set_alpha(1)
        parts['cmeans'].set_facecolor(hex_color)
        parts['cmeans'].set_edgecolor('black')

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)

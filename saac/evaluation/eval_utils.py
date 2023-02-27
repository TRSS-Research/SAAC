import os
import shutil

import numpy as np
import pandas as pd
import pathlib
import warnings
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns

from saac.prompt_generation.prompt_utils import PROMPT_GENERATION_DATA_DIR
from saac.image_analysis.process import ANALYSIS_DIR

EVAL_DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def load_occupation_data(occupation_file=None):
    if occupation_file is None or not os.path.exists(occupation_file):
        occupation_file = os.path.join(PROMPT_GENERATION_DATA_DIR, 'interim', 'AnnualOccupations_TitleBank.csv')
    occupation_data = pd.read_csv(occupation_file)
    # print('occupation data',occupation_data)
    return occupation_data


def load_tda_data(tda_file=None):
    if tda_file is None or not os.path.exists(tda_file):
        tda_file = os.path.join(PROMPT_GENERATION_DATA_DIR, 'interim', 'TDA_Bank.csv')

    cols = [0, 4, 5, 6]
    colnames = [
        'tda',
        'tda_compound',
        'tda_sentiment_cat',
        'tda_sentiment_val'
    ]
    tda_data = pd.read_csv(tda_file, header=0, usecols=cols, names=colnames)
    # print('tda data',tda_data)
    return tda_data


# https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
def rgb_intensity(rgb_tuple):
    return np.mean(np.dot([0.2126, 0.7152, 0.0722], np.array(rgb_tuple)))


def rgb_sorter(rgb_tuples):
    # return sorted(rgb_tuples, key=lambda x: sum(x), reverse=False)
    return sorted(rgb_tuples, key=lambda x: rgb_intensity(x), reverse=False)


'''
Midjourney has a file name limit of 100 characters,which cuts off a number of the submitted prompts in the file name
In order to merge the results with the generated prompt data
We will parse and extract the base prompt from the image_file column
'''


def load_image_analysis_results(analysis_file=None):
    if analysis_file is None:
        eval_data_path = os.path.join(EVAL_DATA_DIRECTORY, 'raw')
        if len(os.listdir(eval_data_path)) < 1:
            analysis_file = [n for n in os.listdir(os.path.join(ANALYSIS_DIR, 'data')) if
                             os.path.splitext(n)[-1] == '.csv']
            print(analysis_file)
            if len(analysis_file) > 0:
                pathlib.Path(eval_data_path).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src=os.path.join(ANALYSIS_DIR, 'data', analysis_file[0]), dst=eval_data_path)

        files = glob.glob(os.path.join(eval_data_path, '*.csv'))
    else:
        files = [analysis_file]
    colnames = ['prompt', 'image', 'quadrant', 'bbox', 'skin color', 'gender.Woman', 'gender.Man']
    results = pd.concat([pd.read_csv(fp, header=0, names=colnames) \
                        .assign(model=os.path.basename(fp).split('_')[0]) for fp in files], sort=False)
    # base_prompt= []
    # for row in results['']:
    #     try:
    #         base_prompt.append(row.split('_')[1:-2])
    #     except:
    #         base_prompt.append('')
    # results['prompt'] = base_prompt
    results['prompt'] = results['prompt'].apply(
        lambda x: 'a ' + x + " photorealistic" if x[0] != 'a' and 'photorealistic' not in x else x)
    # results['prompt'] = results['prompt'].astype(str).str.strip().str.rstrip()
    # Normalizing gender categories
    results['gender.Woman'] = results['gender.Woman'].apply(lambda x: x / 100.)
    results['gender.Man'] = results['gender.Man'].apply(lambda x: x / 100.)
    # Mapping gender detection values to single column
    noface = (results['skin color'].isnull()).values
    unknown = ((results['gender.Woman'] <= .50) & (results['gender.Man'] <= .50)).values
    woman = ((results['gender.Woman'] >= .50) & (results['gender.Man'] < .50)).values
    man = ((results['gender.Man'] >= .50) & (results['gender.Woman'] < .50)).values

    results['gender_detected_cat'] = 0
    results['gender_detected_cat'][noface] = 1
    results['gender_detected_cat'][unknown] = 2
    results['gender_detected_cat'][woman] = 3
    results['gender_detected_cat'][man] = 4
    gender_dict = {
        1: 'no face',
        2: 'unknown',
        3: 'woman',
        4: 'man'
    }
    results['gender_detected_val'] = results['gender_detected_cat'].map(gender_dict)
    # Extracting RGB intensity from skin color
    results['rgb_intensity'] = results['skin color'].apply(lambda x: rgb_intensity(eval(x)) if not pd.isna(x) else None)
    return results


# %%
def load_prompts():
    prompt_data_path = os.path.join(PROMPT_GENERATION_DATA_DIR, 'processed')
    files = glob.glob(os.path.join(prompt_data_path, '*.csv'))
    cols = [0, 1, 5]
    colnames = ['prompt', 'tag', 'compound']
    prompts = pd.concat([pd.read_csv(fp, header=0, usecols=cols, names=colnames) for fp in files], sort=False)
    base_prompt = []
    for line in prompts['prompt']:
        if '/imagine prompt:' in line:  # "/imagine prompt:a pitiless person, photorealistic --s 625"
            line = line.replace('/imagine prompt:', '')
            line = line.replace(', photorealistic --s 625', ' photorealistic')
            line = line.replace(',', ' ')
            line = re.sub(' +', ' ', line).strip().rstrip()
            base_prompt.append(line)
        else:  # "a passionless person, photorealistic"
            line = line.replace(',', ' ')
            line = re.sub(' +', ' ', line).strip().rstrip()
            base_prompt.append(line)
    prompts['prompt'] = base_prompt
    return prompts


def get_occupation_results(res_prompts):
    occupations = load_occupation_data()
    occ_results = res_prompts.merge(occupations, right_on=['norm_title'], left_on=['tag'], how='inner')
    return occ_results


def get_tda_results(res_prompts):
    tda_data = load_tda_data()
    tda_results = res_prompts.merge(tda_data, right_on=['tda'], left_on=['tag'], how='inner')
    return tda_results


def process_analysis(analysis_path=None, savepath=None):
    if savepath is None or not os.path.isdir(savepath):
        savepath = os.path.join(EVAL_DATA_DIRECTORY, 'processed')
        pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
    image_analysis = load_image_analysis_results(analysis_path)
    # print(image_analysis)
    # prompt,image,quadrant,bbox,skin color,gender.Woman,gender.Man
    prompts = load_prompts()
    # print(prompts)
    # prompt,tag,neg,neu,pos,compound
    res_prompts = image_analysis.merge(prompts, right_on=['prompt'], left_on=['prompt'])
    tda = get_tda_results(res_prompts)
    occ = get_occupation_results(res_prompts)
    # print(tda)
    # print(occ)
    pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
    tda.to_csv(os.path.join(savepath, 'TDA_Results.csv'), index=False)
    occ.to_csv(os.path.join(savepath, 'Occupation_Results.csv'), index=False)


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


if __name__ == '__main__':
    warnings.filterwarnings('once')
    process_analysis()

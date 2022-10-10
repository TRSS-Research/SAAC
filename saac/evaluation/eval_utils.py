import os
import numpy as np
import pandas as pd
import warnings
import glob
import re
from saac.prompt_generation.prompt_utils import PROMPT_GENERATION_DATA_DIR

EVAL_DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')

def load_occupation_data(occupation_file=None):
    if occupation_file is None or not os.path.exists(occupation_file):
        occupation_file = os.path.join(PROMPT_GENERATION_DATA_DIR, 'interim', 'AnnualOccupations_TitleBank.csv')
    occupation_data = pd.read_csv(occupation_file)
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
def load_image_analysis_results():
    eval_data_path = os.path.join(EVAL_DATA_DIRECTORY,'raw')
    files = glob.glob(os.path.join(eval_data_path, '*.csv'))
    colnames = ['file_prompt','image_file','quadrant','bbox','skincolor','gender_woman','gender_man']
    results = pd.concat([pd.read_csv(fp,header=0, names=colnames)\
                     .assign(model =os.path.basename(fp).split('_')[0]) for fp in files],sort=False)
    base_prompt= []
    for row in results['image_file']:
        try:
            base_prompt.append(row.split('_')[1:-2])
        except:
            base_prompt.append('')
    results['base_prompt'] = base_prompt
    results['base_prompt'] = results['base_prompt'].apply(lambda x: ' '.join(x)+" photorealistic")
    results['base_prompt'] = results['base_prompt'].str.strip().str.rstrip()
    #Normalizing gender categories
    results['gender_woman'] = results['gender_woman'].apply(lambda x: x / 100.)
    results['gender_man'] = results['gender_man'].apply(lambda x: x / 100.)
    #Mapping gender detection values to single column
    noface= (results['skincolor'].isnull()).values
    unknown=  ((results['gender_woman']<=.50) & (results['gender_man']<=.50)).values
    woman= ((results['gender_woman']>=.50) & (results['gender_man']<.50)).values
    man= ((results['gender_man']>=.50) & (results['gender_woman']<.50)).values

    results['gender_detected_cat'] = 0
    results['gender_detected_cat'][noface]=1
    results['gender_detected_cat'][unknown]=2
    results['gender_detected_cat'][woman]=3
    results['gender_detected_cat'][man]=4
    gender_dict=  {
                 1: 'no face',
                 2: 'unknown',
                 3: 'woman',
                 4: 'man'
                    }
    results['gender_detected_val'] = results['gender_detected_cat'].map(gender_dict)
    #Extracting RGB intensity from skincolor
    results['rgb_intensity'] = results['skincolor'].apply(lambda x:rgb_intensity(eval(x)) if not pd.isna(x) else None)
    return results
#%%
def load_prompts():
    prompt_data_path = os.path.join(PROMPT_GENERATION_DATA_DIR,'processed')
    files = glob.glob(os.path.join(prompt_data_path, '*.csv'))
    cols = [0, 1, 5]
    colnames = ['full_prompt','tag','prompt_compound']
    prompts = pd.concat([pd.read_csv(fp,header=0,usecols= cols, names=colnames)for fp in files],sort=False)
    base_prompt =[]
    for line in prompts['full_prompt']:
        if '/imagine prompt:' in line:
            line = line.replace('/imagine prompt:','')
            line = line.replace(', photorealistic --s 625',' photorealistic')
            line = line.replace(',',' ')
            line = re.sub(' +',' ',line).strip().rstrip()
            base_prompt.append(line)
        else:
            base_prompt.append(line)
    prompts['base_prompt'] = base_prompt
    return prompts

def get_occupation_results():
    results = load_image_analysis_results()
    prompts = load_prompts()
    res_prompts = results.merge(prompts, right_on=['base_prompt'], left_on=['base_prompt'])
    occupations = load_occupation_data()
    occ_results = res_prompts.merge(occupations,right_on=['norm_title'],left_on=['tag'],how='inner')
    return occ_results

def get_tda_results():
    results = load_image_analysis_results()
    prompts = load_prompts()
    res_prompts = results.merge(prompts, right_on=['base_prompt'], left_on=['base_prompt'])
    tda_data = load_tda_data()
    tda_results = res_prompts.merge(tda_data,right_on=['tda'],left_on=['tag'],how='inner')
    return tda_results

def process_analysis(savepath=None):
    if savepath is None or not os.path.isdir(savepath):
        savepath = os.path.join(EVAL_DATA_DIRECTORY,'processed')
    tda = get_tda_results()
    occ = get_occupation_results()
    print(tda,occ)
    tda.to_csv(os.path.join(savepath,'TDA_Results.csv'), index=False)
    occ.to_csv(os.path.join(savepath,'Occupation_Results.csv'), index=False)

if __name__ == '__main__':
    warnings.filterwarnings('once')
    process_analysis()
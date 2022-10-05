import pandas as pd
import re
from textblob import Word
from prompt_utils import score_sentiment
import warnings
warnings.filterwarnings('once')


readpath='../../data/text_generation/raw/'
savepath='../../data/text_generation/interim/'

'''
*  = indicates that a wage estimate is not available
#  = indicates a wage equal to or greater than $100.00 per hour or $208,000 per year 
'''
cols = [
        'OCC_CODE',
        'OCC_TITLE',
        'TOT_EMP',
        'EMP_PRSE',
        'A_MEAN',
        'MEAN_PRSE',
        'A_PCT10',
        'A_PCT25',
        'A_MEDIAN',
        'A_PCT75',
        'A_PCT90'
        ]
dtype_dic = {'OCC_CODE': str, 'OCC_TITLE': str}

jt = pd.read_csv(readpath + 'OEWS21_OccupationsDetailedView.csv',
                 usecols=cols,
                 dtype=dtype_dic,
                 na_values=r"*",
                 keep_default_na=True)
jt.columns = jt.columns.str.lower()

jt = jt[~jt['a_mean'].isnull()]

thous_cols = [
             'tot_emp',
             'a_mean',
             'a_pct10',
             'a_pct25',
             'a_median',
             'a_pct75',
             'a_pct90'
             ]

jt = jt.apply(lambda x: x.str.replace('#', '208000') if x.name in thous_cols else x)
jt = jt.apply(lambda x: x.str.replace(',', '') if x.name in thous_cols else x)
jt = jt.apply(lambda x: x.fillna('0') if x.name in thous_cols else x)
jt = jt.apply(lambda x: x.astype(float) if x.name in thous_cols else x)

pats = [
    ' or ',
    'and',
    'except',
    '/'
    ]

pattern = '|'.join(pats)

jt_norm = jt[~jt['occ_title'].str.contains(pattern, case=False)]
print(len(jt_norm))
jt_dir = jt[jt['occ_title'].str.contains(pattern, case=False)]
print(len(jt_dir))


def clean_string(string):
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'", '"']
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = re.sub(', all other', '', string)
    string = re.sub('helpers--', 'assistant ', string)
    string = re.sub(', general', '', string)
    string = Word(string).singularize()
    string = string.strip()
    string = string.rstrip()
    string = re.sub(' +', ' ', string).strip()
    return string


jt_normv = score_sentiment(jt_norm, 'norm_title', verbose=True)

vlow = jt_normv.loc[jt_normv['a_median'] <= 35000.0]
print(len(vlow))
low = jt_normv.loc[(jt_normv['a_median'] > 35000.0) & (jt_normv['a_median'] <= 50000.0)]
print(len(low))
medium = jt_normv.loc[(jt_normv['a_median'] > 50000.0) & (jt_normv['a_median'] <= 80000.0)]
print(len(medium))
high = jt_normv.loc[(jt_normv['a_median'] > 80000.0) & (jt_normv['a_median'] <= 105000.0)]
print(len(high))
vhigh = jt_normv.loc[jt_normv['a_median'] > 105000.0]
print(len(vhigh))

vlow = (jt_normv['a_median'] <= 35000.0).values
low = ((jt_normv['a_median'] > 35000.0) & (jt_normv['a_median'] <= 50000.0)).values
middle = ((jt_normv['a_median'] > 50000.0) & (jt_normv['a_median'] <= 80000.0)).values
high = ((jt_normv['a_median'] > 80000.0) & (jt_normv['a_median'] <= 105000.0)).values
vhigh = (jt_normv['a_median'] > 105000.0).values

jt_normv['wage_cat'] = 0
jt_normv['wage_cat'][vlow] = 1
jt_normv['wage_cat'][low] = 2
jt_normv['wage_cat'][middle] = 3
jt_normv['wage_cat'][high] = 4
jt_normv['wage_cat'][vhigh] = 5

jt_normv.to_csv(savepath + 'AnnualOccupations_TitleBank.csv', index=False)

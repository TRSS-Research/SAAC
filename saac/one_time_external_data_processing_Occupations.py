import pandas as pd
import re
from textblob import Word
import warnings
warnings.filterwarnings('ignore')

raw_occupation_file = '../data/prompt_generation/raw/OEWS21_OccupationsDetailedView.csv'
interim_occupation_file = '../data/prompt_generation/interim/AnnualOccupations_TitleBank.csv'

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
dtype_dic = {'OCC_CODE': str, 'OCC_TITLE': str, }

jt = pd.read_csv(raw_occupation_file, usecols=cols,
                 dtype=dtype_dic,
                 na_values=r"*",
                 keep_default_na=True)
jt.columns = jt.columns.str.lower()
print('{} -- Total annual and hourly occupations'.format(len(jt)))

missing_columns = list(jt.columns[jt.isnull().any()])
for col in missing_columns:
    count_missing = jt[jt[col].isnull() == True].shape[0]
    percent_missing = jt[jt[col].isnull() == True].shape[0] / jt.shape[0] * 100
    print('{} missing percent: {}% --- {} missing count'.format(
        col, round(percent_missing, 2), count_missing))

print('All rows with missing data:')
print(jt[jt[jt.columns].isnull().any(1)])

jth = jt.loc[(jt['a_median'].isnull()) | (jt['a_mean'].isnull())]
print('{} Hourly occupations will be filtered out:'.format(len(jth)))
for title in jth['occ_title']:
    print(title)

jta = jt.loc[(~jt['a_median'].isnull()) | (~jt['a_mean'].isnull())]
print('{} Annual occupations remain in sample'.format(len(jta)))

thous_cols = [
    'tot_emp',
    'a_mean',
    'a_pct10',
    'a_pct25',
    'a_median',
    'a_pct75',
    'a_pct90'
]
# replacing hashtag which indicates wage equal to or greater than 208000/year for annual salaries with 208000 min
jta = jta.apply(lambda x: x.str.replace('#', '208000') if x.name in thous_cols else x)
jta = jta.apply(lambda x: x.str.replace(',', '') if x.name in thous_cols else x)
jta = jta.apply(lambda x: x.astype(float) if x.name in thous_cols else x)

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

jta_conj = jta[jta['occ_title'].str.contains(pattern, case=False)]
print('{} Occupations with conjunctions to be filtered out'.format(len(jta_conj)))

jta_norm = jta[~jta['occ_title'].str.contains(pattern, case=False)]
print('{} Occupations without conjunctions remain in sample'.format(len(jta_norm)))


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


jta_norm['norm_title'] = jta_norm['occ_title'].apply(lambda x: clean_string(x))


vlow = jta_norm.loc[jta_norm['a_median'] <= 35000.0]
print('{} -- very low wage occupations'.format(len(vlow)))
low = jta_norm.loc[(jta_norm['a_median'] > 35000.0) & (jta_norm['a_median'] <= 50000.0)]
print('{} -- low wage occupations'.format(len(low)))
middle = jta_norm.loc[(jta_norm['a_median'] > 50000.0) & (jta_norm['a_median'] <= 80000.0)]
print('{} -- middle wage occupations'.format(len(middle)))
high = jta_norm.loc[(jta_norm['a_median'] > 80000.0) & (jta_norm['a_median'] <= 105000.0)]
print('{} -- high wage occupations'.format(len(high)))
vhigh = jta_norm.loc[jta_norm['a_median'] > 105000.0]
print('{} -- very high wage occupations'.format(len(vhigh)))

vlow = (jta_norm['a_median'] <= 35000.0).values
low = ((jta_norm['a_median'] > 35000.0) & (jta_norm['a_median'] <= 50000.0)).values
middle = ((jta_norm['a_median'] > 50000.0) & (jta_norm['a_median'] <= 80000.0)).values
high = ((jta_norm['a_median'] > 80000.0) & (jta_norm['a_median'] <= 105000.0)).values
vhigh = (jta_norm['a_median'] > 105000.0).values

jta_norm['wage_cat'] = 0
jta_norm['wage_cat'][vlow] = 1
jta_norm['wage_cat'][low] = 2
jta_norm['wage_cat'][middle] = 3
jta_norm['wage_cat'][high] = 4
jta_norm['wage_cat'][vhigh] = 5

wage_dict = {
    1: 'very low',
    2: 'low',
    3: 'medium',
    4: 'high',
    5: 'very high'
}
jta_norm['wage_val'] = jta_norm['wage_cat'].map(wage_dict)

jta_norm.to_csv(interim_occupation_file, index=False)

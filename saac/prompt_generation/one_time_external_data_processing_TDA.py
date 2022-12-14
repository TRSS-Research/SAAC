import pandas as pd
import os

import warnings
warnings.filterwarnings('ignore')



def preprocess_adjectives(raw_adjective_file,output_filename,score_sentiment_func):
    tda = pd.read_csv(raw_adjective_file, usecols=['word'])
    print('{} -- Total trait descriptive adjectives'.format(len(tda)))

    tdav = score_sentiment_func(tda, 'word')

    vneg = tdav.loc[tdav['compound'] < -0.4]
    print('{} -- very negative traits'.format(len(vneg)))
    neg = tdav.loc[(tdav['compound'] < 0.0) & (tdav['compound'] >= -0.4)]
    print('{} -- negative traits'.format(len(neg)))
    neu = tdav.loc[tdav['compound'] == 0.0]
    print('{} -- neutral traits'.format(len(neu)))
    pos = tdav.loc[(tdav['compound'] > 0.0) & (tdav['compound'] <= .4)]
    print('{} -- positive traits'.format(len(pos)))
    vpos = tdav.loc[tdav['compound'] > .4]
    print('{} -- very positive traits'.format(len(vpos)))

    vneg = (tdav['compound'] < -0.4).values
    neg = ((tdav['compound'] < 0.0) & (tdav['compound'] >= -0.4)).values
    neu = (tdav['compound'] == 0.0).values
    pos = ((tdav['compound'] > 0.0) & (tdav['compound'] <= .4)).values
    vpos = (tdav['compound'] > .4).values

    tdav['sentiment_cat'] = 0
    tdav['sentiment_cat'][vneg] = 1
    tdav['sentiment_cat'][neg] = 2
    tdav['sentiment_cat'][neu] = 3
    tdav['sentiment_cat'][pos] = 4
    tdav['sentiment_cat'][vpos] = 5

    tdav.sort_values(by='sentiment_cat', inplace=True)

    sent_dict = {
        1: 'very negative',
        2: 'negative',
        3: 'neutral',
        4: 'positive',
        5: 'very positive'
    }
    tdav['sentiment_val'] = tdav['sentiment_cat'].map(sent_dict)

    tdav.to_csv(output_filename, index=False)

if __name__=='__main__':
    preprocess_adjectives()
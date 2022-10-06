import pandas as pd
from prompt_utils import score_sentiment
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('once')

raw_adjective_file = '../../data/text_generation/raw/TraitDescriptiveAdjectives.csv'
interim_adjective_file = '../../data/text_generation/interim/TDA_Bank.csv'

tda = pd.read_csv(raw_adjective_file, usecols=['word'])

tdav = score_sentiment(tda, 'word', verbose=True)

tdav.compound.hist()

# Plot compound sentiment
tdav['compound'].plot(marker='o', linewidth=.2, label='trait word')
plt.ylabel('Compound Score')
# Plot line for average compound score
avg_tda = tdav['compound'].mean()
plt.hlines(avg_tda, 0, len(tdav), linewidth=.5, linestyle='dotted', color='green')
plt.legend(title="Trait Descriptive Adjectives", loc='center left', bbox_to_anchor=(1, .9), fancybox=True, shadow=True,)

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

neu = (tdav['compound'] == 0.0).values
pos = ((tdav['compound'] > 0.0) & (tdav['compound'] <= .4)).values
vpos = (tdav['compound'] > .4).values
neg = ((tdav['compound'] < 0.0) & (tdav['compound'] >= -0.4)).values
vneg = (tdav['compound'] < -0.4).values

tdav['sentiment_cat'] = 0
tdav['sentiment_cat'][vneg] = 1
tdav['sentiment_cat'][neg] = 2
tdav['sentiment_cat'][neu] = 3
tdav['sentiment_cat'][pos] = 4
tdav['sentiment_cat'][vpos] = 5

sent_dict = {
    1: 'very negative',
    2: 'negative',
    3: 'neutral',
    4: 'positive',
    5: 'very positive'
}
tdav['sentiment_val'] = tdav['sentiment_cat'].map(sent_dict)

tdav.sort_values(by='sentiment_cat', inplace=True)

tdav.to_csv(interim_adjective_file, index=False)

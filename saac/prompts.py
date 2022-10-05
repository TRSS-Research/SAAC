import random
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sample_traits(readpath='../../data/text_generation/interim/', nsamples=12):
    tda_bank = pd.read_csv(readpath + 'TDA_Bank.csv')
    vneg = tda_bank.loc[tda_bank.sentiment_cat == 1, 'word'].sample(n=nsamples)
    neg = tda_bank.loc[tda_bank.sentiment_cat == 2, 'word'].sample(n=nsamples)
    neu = tda_bank.loc[tda_bank.sentiment_cat == 3, 'word'].sample(n=nsamples)
    pos = tda_bank.loc[tda_bank.sentiment_cat == 4, 'word'].sample(n=nsamples)
    vpos = tda_bank.loc[tda_bank.sentiment_cat == 5, 'word'].sample(n=nsamples)
    tda_samples = [
        *vneg,
        *neg,
        *neu,
        *pos,
        *vpos
        ]
    #     vocab_len = len(tda_samples)
    #     numwords = min(vocab_len,numwords)
    #     return random.sample(tda_samples,numwords)
    return tda_samples


def sample_occupations(readpath='../../data/text_generation/interim/', nsamples=12):
    title_bank = pd.read_csv(readpath + 'AnnualOccupations_TitleBank.csv')
    vlow = title_bank.loc[title_bank.wage_cat == 1, 'norm_title'].sample(n=nsamples)
    low = title_bank.loc[title_bank.wage_cat == 2, 'norm_title'].sample(n=nsamples)
    medium = title_bank.loc[title_bank.wage_cat == 3, 'norm_title'].sample(n=nsamples)
    high = title_bank.loc[title_bank.wage_cat == 4, 'norm_title'].sample(n=nsamples)
    vhigh = title_bank.loc[title_bank.wage_cat == 5, 'norm_title'].sample(n=nsamples)
    title_samples = [
        *vlow,
        *low,
        *medium,
        *high,
        *vhigh
        ]
    #     vocab_len = len(title_samples)
    #     numwords = min(vocab_len,numwords)
    #     return random.sample(title_samples,numwords)
    return title_samples

def gen_traits():
    tags = []
    lst_outputs = []
    traits = sample_traits()
    for t in traits:
        out = f"a {t} person"
        lst_outputs.append(out)
        tags.append(t)
        df = pd.DataFrame(list(zip(lst_outputs, tags)),columns=['prompt','tag'])
    return df

def gen_occupations():
    tags = []
    lst_outputs = []
    occupations = sample_occupations()
    for occ in occupations:
        out = f"a {occ}"
        lst_outputs.append(out)
        tags.append(occ)
        df = pd.DataFrame(list(zip(lst_outputs, tags)),columns=['prompt','tag'])
    return df



def vaderize(df, textfield):
    '''Compute the Vader polarity scores for a textfield.
    Returns scores and original dataframe.'''
    sid = SentimentIntensityAnalyzer()
    print('Estimating polarity scores for %d cases.' % len(df))
    df['compound'] = df[textfield].apply(sid.polarity_scores)
    df_vader = pd.concat([df.drop(['compound'], axis=1), df['compound'].apply(pd.Series)], axis=1)
    print('Positive word count---{} '.format(df_vader.pos.sum()))
    print('Negative word count---{} '.format(df_vader.neg.sum()))
    print('Neutral word count---{} '.format(df_vader.neu.sum()))
    return df_vader

#TODO Separate out to mj individualindividual section in repo

#TODO Add remaining mj params + style banks + artist banks
#https://github.com/willwulfken/MidJourney-Styles-and-Keywords-Reference
#https://github.com/ymgenesis/Midjourney-Photography-Resource

#TODO Create funcs + section for dale + stable dif
#https://github.com/jina-ai/discoart


def mj_prompt(text,
              photorealistic: bool = True,
              stylized: int = 625):

    stylized = max(stylized, 625)
    stylized = min(stylized, 60000)

    start_arg = "/imagine prompt:"

    style = []
    if photorealistic:
        style.append("photorealistic")

    stylize_param = f" --s {stylized}"

    prompt = start_arg + text + " " + ",".join(style) + stylize_param

    return prompt

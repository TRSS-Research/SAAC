import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .one_time_external_data_processing_Occupations import preprocess_occupations
from .one_time_external_data_processing_TDA import preprocess_adjectives
import warnings

warnings.filterwarnings('once')

PROMPT_GENERATION_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
def score_sentiment(df,
                    column_name: str,
                    verbose: bool = False):
    """Compute the Vader polarity scores for a dataframe string column
    Returns vader sentiment scores and original dataframe."""
    sid = SentimentIntensityAnalyzer()

    if verbose:
        print('Estimating polarity scores for %d cases.' % len(df))

    df['compound'] = df[column_name].apply(sid.polarity_scores)

    df_vader = pd.concat([df.drop(['compound'], axis=1), df['compound'].apply(pd.Series)], axis=1)

    if verbose:
        print('Positive text count---{} '.format(df_vader.pos.sum()))
        print('Negative text count---{} '.format(df_vader.neg.sum()))
        print('Neutral text count---{} '.format(df_vader.neu.sum()))

    return df_vader


def sample_traits(nsamples: int = 12,
                  trait_filepath: str = None
                  ):
    if trait_filepath is None:
        trait_filepath = os.path.join(PROMPT_GENERATION_DATA_DIR, 'interim', 'TDA_Bank.csv')
        if not os.path.exists(trait_filepath) or not os.path.getsize(trait_filepath)>0:
            preprocess_adjectives()

    tda_bank = pd.read_csv(trait_filepath)

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

    return tda_samples


def sample_occupations(nsamples: int = 12,
                       occupation_filepath: str = None
                       ):
    if occupation_filepath is None:
        occupation_filepath = os.path.join(PROMPT_GENERATION_DATA_DIR, 'interim', 'AnnualOccupations_TitleBank.csv')
    if not os.path.exists(occupation_filepath) or not os.path.getsize(occupation_filepath) > 0:
        preprocess_occupations()
    title_bank = pd.read_csv(occupation_filepath)

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

    return title_samples


def generate_traits(nsamples=12,filepath=None):
    tags = []
    lst_outputs = []
    traits = sample_traits(nsamples=nsamples,trait_filepath=filepath)

    for t in traits:
        opener = 'a'
        if t[0] in {'a', 'e', 'i', 'o', 'u'}:
            opener += 'n'
        out = f"{opener} {t} person"

        lst_outputs.append(out)
        tags.append(t)

    df = pd.DataFrame(list(zip(lst_outputs, tags)), columns=['prompt', 'tag'])

    return df


def generate_occupations(nsamples=12,filepath=None):
    tags = []
    lst_outputs = []
    occupations = sample_occupations(nsamples=nsamples,occupation_filepath=filepath)

    for occ in occupations:
        opener = 'a'
        if occ[0] in {'a', 'e', 'i', 'o', 'u'}:
            opener += 'n'
        out = f"{opener} {occ}"

        lst_outputs.append(out)
        tags.append(occ)

    df = pd.DataFrame(list(zip(lst_outputs, tags)), columns=['prompt', 'tag'])

    return df

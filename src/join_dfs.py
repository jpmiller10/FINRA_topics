import pandas as pd
import numpy as np
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from tabulate import tabulate
import matplotlib.pyplot as plt 
from features import remove_punctuation

if __name__ == '__main__':
    np.random.seed(10)
    df_arb_mod = pd.read_pickle('data/finra_pickled_arb_model')
    df_arb_mod = df_arb_mod.drop(df_arb_mod[df_arb_mod['NAME'].map(len) < 3].index) 
    df_arb_mod = df_arb_mod.drop(df_arb_mod[df_arb_mod['NAME'].map(len) > 39].index) 
    df_arb_mod = df_arb_mod[~df_arb_mod.NAME.str.contains("NAME")] 
    df_arb_mod['FULL NAME'] = df_arb_mod['NAME']
    df_arb_mod['NAME'] = df_arb_mod['NAME'].apply(lambda x: remove_punctuation(x, punctuation))
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('JD','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('CFP','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('CPA','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('III','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('II','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Jr','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Mr','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Ms','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Mrs','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Dr','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Honorable','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Hon','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Judge','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Professor','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Ph D','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('J D','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('MBA','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('Sr','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('PE','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace('ill','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].replace(' ','', regex=True)
    df_arb_mod['NAME'] = df_arb_mod['NAME'].apply(lambda x: str(x).lower())
    df_arb_mod['arbitrator'] = df_arb_mod['NAME']

    df_arbitrations = pd.read_pickle('data/finra_pickled_arbitrators')

    df_joined = df_arbitrations.join(df_arb_mod.set_index('arbitrator'), on = 'arbitrator')

    df_joined.to_pickle("data/finra_joined")
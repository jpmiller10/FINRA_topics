import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from string import punctuation
from tabulate import tabulate
import features

def load_csv(file_path):
    '''
    Creates dataframe for cleaning   
    Arguments:
        file_path: filepath for the .csv 
    Returns:
        df: dataframe
    '''
    df= pd.read_csv(file_path, delimiter=',', lineterminator='|')
    return df

def keep_cols(df, cols):
    '''
    Returns df with specified columns
    
    ARGS:
        df - pd.dataFrame
        cols - list of columns
    '''
    columns_to_drop = []
    for i in df.columns:
        if i not in cols:
            columns_to_drop.append(i)
    df.drop(columns_to_drop, inplace=True, axis=1)
    return df

def strip_nl(columns):
    pass

def get_arbs(column):
    #returns only arbitrations based on url https://www.finra.org/sites/default/files/aao_documents/
    pass

def topic_counts(df):
    #counts of each topic in grouped columns of df
    grouped = df[['topics','Background Information']].groupby(['topics']).count().sort_values(by = 'Background Information',ascending = False)
    print(tabulate(grouped.head(10), headers='keys', tablefmt='github'))

if __name__ == "__main__":
    df_arb = load_csv('data/advlaw_arbs.csv')

    cols_to_keep = ['Arbitrator ID', 'NAME', 'CRD', 'City/State/Country',
       'Statutory Discrimination Qualified', 'Classification', 'Chair Status',
       'Start Date', 'End Date', 'Firm', 'Position', 'Non-Inv-Related',
       'Disc Outline Activities', 'Public Cases', 'Non Public Cases',
       'Current Cases', 'Chair Count', 'Panelist Count',
       'Background Information', 'Expungement Count']
    
    df_arb = keep_cols(df_arb, cols_to_keep)

    df_arb= df_arb.replace('\r\n','', regex=True)
    df_arb = df_arb.drop_duplicates(subset = ['Arbitrator ID'], keep = 'first') 

    cols_for_model = ['Arbitrator ID', 'NAME', 'Chair Count', 'Panelist Count',
       'Background Information', 'City/State/Country']

    df_arb_mod = keep_cols(df_arb, cols_for_model)

    additional_stop_words = [
        "the",
        "raymond",
        "james",
        "alleged",
        "involvedaccount",
        "involveduse"]
    
    stop_words = features.get_stop_words(additional_stop_words)
    punc = punctuation
    n_topics = 10
    n_top_words = 10
    features.clean_column(df_arb_mod, 'Background Information', punc)

    X, feature = features.vectorize(df_arb_mod, 'Background Information', stop_words)
    W, H = features.get_nmf(X, n_components=n_topics)
    top_words = features.get_topic_words(H, feature, n_features=n_top_words)
    df_arb_mod['topics'] = features.document_topics(W)
    #df_firna['topics'] = df_firna['topics']
    features.print_topics(top_words)
    df_arb_mod.to_pickle("data/finra_pickled_arb_model")
    df_arb.to_pickle("data/finra_pickled_arb")
    topic_counts(df_arb_mod)

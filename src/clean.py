import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

def load_csv(file_path):
    '''
    Creates dataframe for cleaning   
    Arguments:
        file_path: filepath for the .csv 
    Returns:
        df: dataframe
    '''
    df= pd.read_csv(file_path)
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

def to_datetime(df,cols):
    '''
    Converts 'col' to datetime     
    Arguments:
        df: dataframe
        cols: a list of columns
    Returns:
        df: dataframe
    '''
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    # time_range_filter = (df[col] > start) & (df[col] < end)
    # df = df[time_range_filter].copy()
    return df

def to_numeric(df, cols):
    '''
    Arguments:
        df: dataframe
        cols: a list of columns
    Retrun:
        df: dataframe
    '''
    for c in cols:
        df[c] = df[c].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    return df


def to_categorical(df, cols):
    '''
    Arguments:
        df: dataframe
        cols: a list of columns
    Retrun:
        df: dataframe
    ''' 
    for c in cols:
        df[c] = df[c].astype("category")
    return df

def fill_resolutions(df, col, str):
    '''
    Arguments:
        col1: str of latitude column name
        col2: str of longitude column name
    Retrun:
        df: dataframe
    ''' 
    df[col] = df[col].fillna(str)  
    return df

def drop_nan_col(df, col):
    '''
    Arguments:
        df: dataframe
        col: column to drop NaNs from
    Retrun:
        df: dataframe
    ''' 
    df= df.dropna(subset=[col], inplace=True) 
    return df

 

if __name__ == '__main__':
    df = load_csv('data/finra_data.csv')
    df = fill_resolutions(df, 'resolution', 'Favorable for Broker') 
    drop_nan_col(df, 'allegations')
    col_list = ['resolution']
    df = to_categorical(df, col_list)
    #to_datetime(df,['date_initiated', 'resolution_date']) - unused
    df['targets_1'] = df['resolution']
    df['targets_1'] = df['targets_1'].replace('Dismissed','Favorable')
    df['targets_1'] = df['targets_1'].replace('Favorable for Broker','Favorable')
    df['targets_1'] = df['targets_1'].replace('Withdrawn','Favorable')
    df['targets_1'] = df['targets_1'].replace('Acceptance, Waiver & Consent(AWC)','Unfavorable')
    df['targets_1'] = df['targets_1'].replace('Consent','Unfavorable')
    df['targets_1'] = df['targets_1'].replace('Decision','Unfavorable')
    df['targets_1'] = df['targets_1'].replace('Decision & Order of Offer of Settlement','Unfavorable')
    df['targets_1'] = df['targets_1'].replace('Judgment Rendered','Unfavorable')
    df['targets_1'] = df['targets_1'].replace('Order','Unfavorable')
    df['targets_1'] = df['targets_1'].replace('Other','Unfavorable')
    df['targets_1'] = df['targets_1'].replace('Settled','Unfavorable')
    df['targets_1'] = df['targets_1'].replace('Stipulation and Consent','Unfavorable')
    to_categorical(df, ['targets_1'])
    df['targets_2'] = df['resolution']
    df['targets_2'] = df['targets_2'].replace('Dismissed','Favorable')
    df['targets_2'] = df['targets_2'].replace('Favorable for Broker','Favorable')
    df['targets_2'] = df['targets_2'].replace('Withdrawn','Favorable')
    df['targets_2'] = df['targets_2'].replace('Acceptance, Waiver & Consent(AWC)','Unfavorable')
    df['targets_2'] = df['targets_2'].replace('Consent','Settled')
    df['targets_2'] = df['targets_2'].replace('Decision','Unfavorable')
    df['targets_2'] = df['targets_2'].replace('Decision & Order of Offer of Settlement','Settled')
    df['targets_2'] = df['targets_2'].replace('Judgment Rendered','Unfavorable')
    df['targets_2'] = df['targets_2'].replace('Order','Unfavorable')
    df['targets_2'] = df['targets_2'].replace('Other','Unfavorable')
    df['targets_2'] = df['targets_2'].replace('Settled','Settled')
    df['targets_2'] = df['targets_2'].replace('Stipulation and Consent','Settled')
    to_categorical(df, ['targets_2'])
    df.to_pickle('data/finra_pickled_df')
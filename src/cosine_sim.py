import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def cos_arbitration_recommendations(arbitrator_name, X, n=5, state=None):
    index = df.index[(df['FULL NAME'] == arbitrator_name)][0]
    trail = X[index].reshape(1,-1)
    cs = cosine_similarity(trail, X)
    rec_index = np.argsort(cs)[0][::-1][1:]
    ordered_df = df.loc[rec_index]
    if state:
        ordered_df = ordered_df[ordered_df['state'] == state]
    rec_df = ordered_df.head(n)
    orig_row = df.loc[[index]].rename(lambda x: 'original')
    total = pd.concat((orig_row,rec_df))
    return total

if __name__ == "__main__":
    
    df = pd.read_pickle('data/finra_joined')
    df = df.reset_index(drop=True)

    df['case_value']= df['case_value']*10
    df['denial'] = df['denial']*1
    df['expunged'] = df['expunged']*1   
    df['case_topics']= df['case_topics']*10
    df['Chair Count'] = df['Chair Count'].fillna(df['Chair Count'].mean())
    df['Chair Count'] = df['Chair Count']*0
    df['Panelist Count'] = df['Panelist Count'].fillna(df['Panelist Count'].mean())
    df['Panelist Count'] = df['Panelist Count']*0
    df['topics'] = df['topics'].fillna(0)
    df['topics'] = df['topics']*1

    df.fillna('Unable to Determine', inplace=True)

    features = ['case_value', 
                'denial', 
                'expunged', 
                'case_topics', 
                'Chair Count', 
                'Panelist Count', 
                'topics']

    X = df[features].values

    ss = StandardScaler()
    X = ss.fit_transform(X)
    
    print(cos_arbitration_recommendations('Mr. Edward A. Trabin',X,n=5,state=None))
    # print(cos_arbitration_recommendations('Mr. Edward A. Trabin','FL',X,n=5,state='AZ'))
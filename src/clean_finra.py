import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from clean import to_numeric
from string import punctuation
from features import remove_punctuation
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
    df= pd.read_csv(file_path, delimiter='|', lineterminator='~')
    return df

def strip_nl(columns):
    pass

# def split_arbs(df, column, splits):
#     columns = [x.lower for x in splits]
#     columns = columns.replace(' ', '_')
#     for i in splits:
        # df['representation']= (df_case['body'].str.split("REPRESENTATION OF PARTIES", n = 1, expand = True))[1]
        # df_case['body']= (df_case['body'].str.split("REPRESENTATION OF PARTIES", n = 1, expand = True))[0]

def get_arbs(column):
    #returns only arbitrations based on url https://www.finra.org/sites/default/files/aao_documents/
    pass

def topic_counts(df):
    #counts of each topic in grouped columns of df
    grouped = df[['case_topics','case_summary']].groupby(['case_topics']).count().sort_values(by = 'case_summary',ascending = False)
    print(tabulate(grouped.head(10), headers='keys', tablefmt='github'))

if __name__ == "__main__":
    # df_case = load_csv('data/finra_5k.txt')
    # df_case = load_csv('data/finra_20k.txt')
    df_case = load_csv('data/scraped_content.txt')
    df_case['site'] = df_case['site'].replace('\n','', regex=True)
    df_case = df_case[df_case.site.str.contains("AAO")] 
    df_case['site'] = df_case['site'].replace('HTTPS://WWW.FINRA.ORG/SITES/DEFAULT/FILES/AAO_DOCUMENTS/','', regex=True)
    df_case['site'] = df_case['site'].replace('.PDF','', regex=True)
    df_case['year'] = df_case['site'].astype(str).str[0:2] 
    df_case['type'] = df_case['body'].astype(str).str[0:5] 
    df_case = df_case[df_case.type.str.contains("Award")] 
  
    df_case['arbitrators']= (df_case['body'].str.rsplit("ARB", n = 1, expand = True))[1]
    df_case['fees']= (df_case['body'].str.rsplit("FEES", n = 1, expand = True))[1]
    df_case['award']= (df_case['body'].str.rsplit("AWARD", n = 1, expand = True))[1]
    df_case['other_issues']= (df_case['body'].str.rsplit("OTHER ISSUES CONSIDERED AND DECIDED", n = 1, expand = True))[1]
    df_case['relief_requested']= (df_case['body'].str.rsplit("RELIEF REQUESTED", n = 1, expand = True))[1]
    df_case['case_summary']= (df_case['body'].str.rsplit("CASE SUMMARY", n = 1, expand = True))[1]
    df_case['representation']= (df_case['body'].str.rsplit("REPRESENTATION OF PARTIES", n = 1, expand = True))[1]

    # df_case['arbitrators']= (df_case['arbitrators'].str.split("ARB", n = 1, expand = True))[1]
    df_case['fees']= (df_case['fees'].str.split("ARB", n = 1, expand = True))[0]
    df_case['award']= (df_case['award'].str.split("FEES", n = 1, expand = True))[0]
    df_case['other_issues']= (df_case['other_issues'].str.split("AWARD", n = 1, expand = True))[0]
    df_case['relief_requested']= (df_case['relief_requested'].str.split("OTHER ISSUES CONSIDERED AND DECIDED", n = 1, expand = True))[0]
    df_case['case_summary']= (df_case['case_summary'].str.split("RELIEF REQUESTED", n = 1, expand = True))[0]
    df_case['representation']= (df_case['representation'].str.split("CASE SUMMARY", n = 1, expand = True))[0]

    df_case = to_numeric(df_case, ['year'])

    df_case = df_case.dropna()
    df_case.reset_index(drop=True, inplace=True) 

    df_arbitrators = (df_case['arbitrators'].str.split('\n', n = -1, expand = True))
    df_case['arbitrator_1']=df_arbitrators[1]
    df_case['arbitrator_2']=df_arbitrators[2]
    df_case['arbitrator_3']=df_arbitrators[3]

    df_case = df_case[df_case.award.str.contains("xpung")]  

    df_case['denial']=(df_case['award'].map(lambda x: True if 'expungement is denied' in str(x) else False)).astype(int)
    df_case['expunged']=(df_case['award'].map(lambda x: True if 'expungement of all' in str(x) else False)).astype(int)

    df_case['case_value']= (df_case['denial']*-1) + (df_case['expunged']*1.5) 
    df_case['body'] = df_case['body'].replace('\n',' ', regex=True)

    df_case.to_pickle("data/finra_pickled_case")

    df_chair = df_case[['arbitrator_1', 'case_value', 'denial', 'expunged', 'site', 'year', 'body', 'case_summary']]
    df_chair = df_chair.rename(columns={'old':'new', 'arbitrator_1': 'arbitrator'})
    df_chair['case_value'] = df_chair['case_value']*1.5
    df_pan1 = df_case[['arbitrator_2', 'case_value', 'denial', 'expunged', 'site', 'year', 'body', 'case_summary']]
    df_pan1 = df_pan1.rename(columns={'old':'new', 'arbitrator_2': 'arbitrator'})
    df_pan2 = df_case[['arbitrator_3', 'case_value', 'denial', 'expunged', 'site', 'year', 'body', 'case_summary']]
    df_pan2 = df_pan2.rename(columns={'old':'new', 'arbitrator_3': 'arbitrator'})

    frames = [df_chair, df_pan1, df_pan2]

    df_arbitrators = pd.concat(frames)
    df_arbitrators = df_arbitrators.drop(df_arbitrators[df_arbitrators['arbitrator'].map(len) < 3].index) 
    df_arbitrators = df_arbitrators.drop(df_arbitrators[df_arbitrators['arbitrator'].map(len) > 39].index) 
    df_arbitrators = df_arbitrators[~df_arbitrators.arbitrator.str.contains("Arbitrator")] 
    df_arbitrators['full_name'] = df_arbitrators['arbitrator']
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].apply(lambda x: remove_punctuation(x, punctuation))
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('JD','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('CFP','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('CPA','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('III','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('II','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Jr','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Mr','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Ms','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Mrs','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Dr','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Honorable','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Hon','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Judge','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Professor','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Ph D','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('J D','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('MBA','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('Sr','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('PE','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace('ill','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].replace(' ','', regex=True)
    df_arbitrators['arbitrator'] = df_arbitrators['arbitrator'].apply(lambda x: str(x).lower())

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
    features.clean_column(df_arbitrators, 'case_summary', punc)

    X, feature = features.vectorize(df_arbitrators, 'case_summary', stop_words)
    W, H = features.get_nmf(X, n_components=n_topics)
    top_words = features.get_topic_words(H, feature, n_features=n_top_words)
    df_arbitrators['case_topics'] = features.document_topics(W)
    #df_firna['topics'] = df_firna['topics']
    features.print_topics(top_words)
    topic_counts(df_arbitrators)


    df_arbitrators.to_pickle("data/finra_pickled_arbitrators")
    # splits = ['REPRESENTATION OF PARTIES', 'CASE SUMMARY', 'RELIEF REQUESTED', 'OTHER ISSUES CONSIDERED AND DECIDED', 'AWARD', 'FEES', 'ARBITRATOR']
 
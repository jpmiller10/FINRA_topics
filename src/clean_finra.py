import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from clean import to_numeric

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

if __name__ == "__main__":
    df_case = load_csv('data/finra_5k.txt')
    df_case['site'] = df_case['site'].replace('\n','', regex=True)
    df_case = df_case[df_case.site.str.contains("AAO")] 
    df_case['site'] = df_case['site'].replace('HTTPS://WWW.FINRA.ORG/SITES/DEFAULT/FILES/AAO_DOCUMENTS/','', regex=True)
    df_case['site'] = df_case['site'].replace('.PDF','', regex=True)
    df_case['year'] = df_case['site'].astype(str).str[0:2] 
    df_case['type'] = df_case['body'].astype(str).str[0:5] 
    df_case = df_case[df_case.type.str.contains("Award")] 
    # df_case['representation']= (df_case['body'].str.split("REPRESENTATION OF PARTIES", n = 1, expand = True))[1]
    # df_case['body']= (df_case['body'].str.split("REPRESENTATION OF PARTIES", n = 1, expand = True))[0]
    # df_case['case_summary']= (df_case['representation'].str.split("CASE SUMMARY", n = 1, expand = True))[1]
    # df_case['representation']= (df_case['body'].str.split("REPRESENTATION OF PARTIES", n = 1, expand = True))[0]
    # df_case['relief_requested']= (df_case['representation'].str.split("RELIEF REQUESTED", n = 1, expand = True))[1]
    # df_case['case_summary']= (df_case['representation'].str.split("CASE SUMMARY", n = 1, expand = True))[0]
    # df_case['other_issues']= (df_case['representation'].str.split("OTHER ISSUES CONSIDERED AND DECIDED", n = 1, expand = True))[1]
    # df_case['relief_requested']= (df_case['representation'].str.split("RELIEF REQUESTED", n = 1, expand = True))[0]
    # df_case['award']= (df_case['representation'].str.split("AWARD", n = 1, expand = True))[1]
    # df_case['other_issues']= (df_case['representation'].str.split("OTHER ISSUES CONSIDERED AND DECIDED", n = 1, expand = True))[0]
    # df_case['fees']= (df_case['representation'].str.split("FEES", n = 1, expand = True))[1]
    # df_case['award']= (df_case['representation'].str.split("AWARD", n = 1, expand = True))[0]
    # df_case['arbitrators']= (df_case['representation'].str.split("ARBIT", n = 1, expand = True))[1]
    # df_case['fees']= (df_case['representation'].str.split("FEES", n = 1, expand = True))[0]

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

    df_chair = df_case[['arbitrator_1', 'case_value', 'denial', 'expunged', 'site', 'year', 'body']]
    df_chair = df_chair.rename(columns={'old':'new', 'arbitrator_1': 'arbitrator'})
    #need to weight
    df_pan1 = df_case[['arbitrator_2', 'case_value', 'denial', 'expunged', 'site', 'year', 'body']]
    df_pan1 = df_pan1.rename(columns={'old':'new', 'arbitrator_2': 'arbitrator'})
    df_pan2 = df_case[['arbitrator_3', 'case_value', 'denial', 'expunged', 'site', 'year', 'body']]
    df_pan2 = df_pan2.rename(columns={'old':'new', 'arbitrator_3': 'arbitrator'})

    # splits = ['REPRESENTATION OF PARTIES', 'CASE SUMMARY', 'RELIEF REQUESTED', 'OTHER ISSUES CONSIDERED AND DECIDED', 'AWARD', 'FEES', 'ARBITRATOR']
 
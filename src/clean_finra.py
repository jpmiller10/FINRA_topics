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
    df= pd.read_csv(file_path, delimiter=',', lineterminator=';')
    return df

if __name__ == "__main__":
    df = load_csv('data/test_scrap.txt')
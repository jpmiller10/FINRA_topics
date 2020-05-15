import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
plt.style.use('tableau-colorblind10')
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}

def print_top_words(categories):
    for cat in range(len(categories)):
        print(f"\nTarget: {cat}, name: {target_names[cat]}")
        log_prob = nb_model.feature_log_prob_[cat]
        i_topn = np.argsort(log_prob)[::-1][:n]
        features_topn = [feature_words[i] for i in i_topn]
        print(f"Top {n} tokens: ", features_topn)

if __name__ == "__main__":
    df_finra = pd.read_pickle('data/finra_pickled_df_top')
    df_finra0 = df_finra.loc[lambda df_finra: df_finra['topics'] == 0, :]
    df_finra1 = df_finra.loc[lambda df_finra: df_finra['topics'] == 1, :]
    df_finra2 = df_finra.loc[lambda df_finra: df_finra['topics'] == 2, :]
    df_finra3 = df_finra.loc[lambda df_finra: df_finra['topics'] == 3, :]
    df_finra4 = df_finra.loc[lambda df_finra: df_finra['topics'] == 4, :]
  
    print("\nTOPIC 0 Misrepresentation/Breach of Fiduciary Duty.")
    categories = ['Favorable', 'Unfavorable']
    y = df_finra0['targets_1']
    X = df_finra0['allegations']
    target_names = categories

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.25, 
                                                random_state=1)

    count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_df=1.0, min_df=1, max_features=None)

    count_vect.fit(X_train);
    X_train_counts = count_vect.transform(X_train)
    
    tfidf_transformer = TfidfTransformer(use_idf=True)
    tfidf_transformer.fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    nb_model.fit(X_train_tfidf, y_train);

    feature_words = count_vect.get_feature_names()
    n = 10 #number of top words associated with the category that we wish to see

    print_top_words(categories)

    nb_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', MultinomialNB()),
                        ])
    nb_pipeline.fit(X_train, y_train); 

    docs_test = X_test
    predicted = nb_pipeline.predict(docs_test)
    accuracy = np.mean(predicted == y_test)
    print("\nThe accuracy on the Misrepresentation/Breach of Fiduciary Duty Topic set is {0:0.3f}.".format(accuracy))

    print("\nTOPIC 1 Misrepresentation/Breach of Fiduciary Duty.")
    categories = ['Favorable', 'Unfavorable']
    y = df_finra1['targets_1']
    X = df_finra1['allegations']
    target_names = categories

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.25, 
                                                random_state=1)

    count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_df=1.0, min_df=1, max_features=None)

    count_vect.fit(X_train);
    X_train_counts = count_vect.transform(X_train)
    
    tfidf_transformer = TfidfTransformer(use_idf=True)
    tfidf_transformer.fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    nb_model.fit(X_train_tfidf, y_train);

    feature_words = count_vect.get_feature_names()
    n = 10 #number of top words associated with the category that we wish to see

    print_top_words(categories)

    nb_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', MultinomialNB()),
                        ])
    nb_pipeline.fit(X_train, y_train); 

    docs_test = X_test
    predicted = nb_pipeline.predict(docs_test)
    accuracy = np.mean(predicted == y_test)
    print("\nThe accuracy on the Trade Violation Topic set is {0:0.3f}.".format(accuracy))
        
    print("\nTOPIC 2 Reporting Violation.")
    categories = ['Favorable', 'Unfavorable']
    y = df_finra2['targets_1']
    X = df_finra2['allegations']
    target_names = categories

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.25, 
                                                random_state=1)

    count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_df=1.0, min_df=1, max_features=None)

    count_vect.fit(X_train);
    X_train_counts = count_vect.transform(X_train)
    
    tfidf_transformer = TfidfTransformer(use_idf=True)
    tfidf_transformer.fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    nb_model.fit(X_train_tfidf, y_train);

    feature_words = count_vect.get_feature_names()
    n = 10 #number of top words associated with the category that we wish to see

    print_top_words(categories)

    nb_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', MultinomialNB()),
                        ])
    nb_pipeline.fit(X_train, y_train); 

    docs_test = X_test
    predicted = nb_pipeline.predict(docs_test)
    accuracy = np.mean(predicted == y_test)
    print("\nThe accuracy on the Reporting Violation Topic set is {0:0.3f}.".format(accuracy))

    print("\nTOPIC 3 Violation of Law.")
    categories = ['Favorable', 'Unfavorable']
    y = df_finra3['targets_1']
    X = df_finra3['allegations']
    target_names = categories

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.25, 
                                                random_state=1)

    count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_df=1.0, min_df=1, max_features=None)

    count_vect.fit(X_train);
    X_train_counts = count_vect.transform(X_train)
    
    tfidf_transformer = TfidfTransformer(use_idf=True)
    tfidf_transformer.fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    nb_model.fit(X_train_tfidf, y_train);

    feature_words = count_vect.get_feature_names()
    n = 10 #number of top words associated with the category that we wish to see

    print_top_words(categories)

    nb_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', MultinomialNB()),
                        ])
    nb_pipeline.fit(X_train, y_train); 

    docs_test = X_test
    predicted = nb_pipeline.predict(docs_test)
    accuracy = np.mean(predicted == y_test)
    print("\nThe accuracy on the Violation of Law Topic set is {0:0.3f}.".format(accuracy))

    print("\nTOPIC 4 Auction Rate Securities.")
    categories = ['Favorable', 'Unfavorable']
    y = df_finra4['targets_1']
    X = df_finra4['allegations']
    target_names = categories

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.25, 
                                                random_state=1)

    count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_df=1.0, min_df=1, max_features=None)

    count_vect.fit(X_train);
    X_train_counts = count_vect.transform(X_train)
    
    tfidf_transformer = TfidfTransformer(use_idf=True)
    tfidf_transformer.fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    nb_model.fit(X_train_tfidf, y_train);

    feature_words = count_vect.get_feature_names()
    n = 10 #number of top words associated with the category that we wish to see

    print_top_words(categories)

    nb_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', MultinomialNB()),
                        ])
    nb_pipeline.fit(X_train, y_train); 

    docs_test = X_test
    predicted = nb_pipeline.predict(docs_test)
    accuracy = np.mean(predicted == y_test)
    print("\nThe accuracy on the Auction Rate Securities Topic set is {0:0.3f}.".format(accuracy))
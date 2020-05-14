import pandas as pd 
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from clean import to_categorical


if __name__ == "__main__":
    df_finra = pd.read_pickle('data/finra_pickled_df_top')
    df_finra['targets'] = df_finra['resolution']
    df_finra['targets'] = df_finra['targets'].replace('Dismissed','Favorable')
    df_finra['targets'] = df_finra['targets'].replace('Favorable for Broker','Favorable')
    df_finra['targets'] = df_finra['targets'].replace('Withdrawn','Favorable')
    df_finra['targets'] = df_finra['targets'].replace('Acceptance, Waiver & Consent(AWC)','Unfavorable')
    df_finra['targets'] = df_finra['targets'].replace('Consent','Unfavorable')
    df_finra['targets'] = df_finra['targets'].replace('Decision','Unfavorable')
    df_finra['targets'] = df_finra['targets'].replace('Decision & Order of Offer of Settlement','Unfavorable')
    df_finra['targets'] = df_finra['targets'].replace('Judgment Rendered','Unfavorable')
    df_finra['targets'] = df_finra['targets'].replace('Order','Unfavorable')
    df_finra['targets'] = df_finra['targets'].replace('Settled','Unfavorable')
    df_finra['targets'] = df_finra['targets'].replace('Stipulation and Consent','Unfavorable')
    to_categorical(df_finra, ['targets'])
   
   
    categories = ['Favorable', 'Unfavorable']

    # Create train & test splits
    y_train_validation = df_finra['targets']
    X_train_validation = df_finra['allegations']
    # y_holdout_test = df_holdout_test.pop('active')
    # X_holdout_test = df_holdout_test.values
    
    # Cross Validation Split
    X_train, X_test, y_train, y_test = train_test_split(X_train_validation,
                                                            y_train_validation, 
                                                            test_size=0.25, 
                                                            random_state=42)

    count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_df=1.0, min_df=1, max_features=None)

    count_vect.fit(X_train)

    target_names = categories

    X_train_counts = count_vect.transform(X_train)
    
    print("The type of X_train_counts is {0}.".format(type(X_train_counts)))
    print("The X matrix has {0} rows (documents) and {1} columns (words).".format(
        X_train_counts.shape[0], X_train_counts.shape[1]))

    tfidf_transformer = TfidfTransformer(use_idf=True)
    tfidf_transformer.fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    nb_model.fit(X_train_tfidf, y_train);

    feature_words = count_vect.get_feature_names()
    n = 7 #number of top words associated with the category that we wish to see

    for cat in range(len(categories)):
        print(f"\nTarget: {cat}, name: {target_names[cat]}")
        log_prob = nb_model.feature_log_prob_[cat]
        i_topn = np.argsort(log_prob)[::-1][:n]
        features_topn = [feature_words[i] for i in i_topn]
        print(f"Top {n} tokens: ", features_topn)

    nb_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', MultinomialNB()),
                        ])
    nb_pipeline.fit(X_train, y_train); 

    # twenty_test = fetch_20newsgroups(subset='test', categories=categories,
    #                                  shuffle=True, random_state=42)
    docs_test = X_test
    predicted = nb_pipeline.predict(docs_test)
    accuracy = np.mean(predicted == y_test)
    print("\nThe accuracy on the test set is {0:0.3f}.".format(accuracy))
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

def define_axis_style(ax, title, x_label, y_label, legend=False):
    '''
    Function to define labels/title font sizes for consistency across plots
    '''
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xlabel(x_label, fontsize=16)
    ax.tick_params(labelsize=14)
    if legend:
        ax.legend(fontsize=16)
    return

def plot_feature_importances(ax, feat_importances, feat_std_deviations, feat_names, n_features, outfilename):
    '''
    Plot feature importances for an NLP model
    feat_importances : Array of feature importances
    feat_std_deviations : Standard deviations of feature importances (intended for RandomForest) **OPTIONAL
    feat_names : Array of feature names
    n_features : Number of top features to include in plot
    outfilename : Path to save file
    '''
    feat_importances = np.array(feat_importances)
    feat_names = np.array(feat_names)
    sort_idx = feat_importances.argsort()[::-1][:n_features]
    if len(feat_std_deviations) > 0:
        feat_std_deviations = feat_std_deviations[sort_idx]
    else:
        feat_std_deviations = None
    ax.barh(feat_names[sort_idx], feat_importances[sort_idx], color='darkgreen', edgecolor='black', linewidth=1, yerr=feat_std_deviations)
    plt.tight_layout()
    plt.savefig(outfilename)
    plt.close('all')
    return

if __name__ == "__main__":
    df_finra = pd.read_pickle('data/finra_pickled_df_top')
    
    print("\nSTART WITH 2 TARGETS.")

    categories = ['Favorable', 'Unfavorable']
    y = df_finra['targets_1']
    X = df_finra['allegations']
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
    print("\nThe accuracy on the Favorable/Unfavorable test set is {0:0.3f}.".format(accuracy))

    # Print and plot feature importances for each class
    target_names = np.unique(y)
    n=15 # number of top words to include
    feature_words = count_vect.get_feature_names()
    for cat in range(len(np.unique(y))):
        print(f"\nTarget: {cat}, name: {target_names[cat]}")
        log_probs= nb_model.feature_log_prob_[cat]
        top_idx = np.argsort(log_probs)[::-1][:n]
        features_top_n = [feature_words[i] for i in top_idx]
        print(f"Top {n} tokens: ", features_top_n)
        plt.style.use('tableau-colorblind10')
        fig, ax = plt.subplots(1, 1, figsize=(10,6))

        # Make it pretty/consistent
        define_axis_style(ax = ax, title=f'Top {n} Features: {target_names[cat]}', y_label=None, x_label="Log Probability")
        plot_feature_importances(
                                ax = ax,
                                feat_importances = nb_model.feature_log_prob_[cat],
                                feat_std_deviations = [],
                                feat_names = feature_words,
                                n_features = n,
                                outfilename = f'images/nb_top_features_2_{target_names[cat]}'
                            )
        
    print("\nNOW LOADING 3 TARGETS.")

    categories = ['Favorable', 'Settled', 'Unfavorable']
    y = df_finra['targets_2']
    X = df_finra['allegations']
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
    print("\nThe accuracy on the Favorable/Settled/Unfavorable test set is {0:0.3f}.".format(accuracy))

    # Print and plot feature importances for each class
    target_names = np.unique(y)
    n=15 # number of top words to include
    feature_words = count_vect.get_feature_names()
    for cat in range(len(np.unique(y))):
        print(f"\nTarget: {cat}, name: {target_names[cat]}")
        log_probs= nb_model.feature_log_prob_[cat]
        top_idx = np.argsort(log_probs)[::-1][:n]
        features_top_n = [feature_words[i] for i in top_idx]
        print(f"Top {n} tokens: ", features_top_n)
        plt.style.use('tableau-colorblind10')
        fig, ax = plt.subplots(1, 1, figsize=(10,6))

        # Make it pretty/consistent
        define_axis_style(ax = ax, title=f'Top {n} Features: {target_names[cat]}', y_label=None, x_label="Log Probability")
        plot_feature_importances(
                                ax = ax,
                                feat_importances = nb_model.feature_log_prob_[cat],
                                feat_std_deviations = [],
                                feat_names = feature_words,
                                n_features = n,
                                outfilename = f'images/nb_top_features_3_{target_names[cat]}'
                            )
        

    print("\nNOW LOADING 12 TARGETS.")

    categories = df_finra['resolution'].unique()
    y = df_finra['resolution']
    X = df_finra['allegations']
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
    print("\nThe accuracy on the Twelve Origional Topics test set is {0:0.3f}.".format(accuracy))

        # Print and plot feature importances for each class
    target_names = np.unique(y)
    n=15 # number of top words to include
    feature_words = count_vect.get_feature_names()
    for cat in range(len(np.unique(y))):
        print(f"\nTarget: {cat}, name: {target_names[cat]}")
        log_probs= nb_model.feature_log_prob_[cat]
        top_idx = np.argsort(log_probs)[::-1][:n]
        features_top_n = [feature_words[i] for i in top_idx]
        print(f"Top {n} tokens: ", features_top_n)
        plt.style.use('tableau-colorblind10')
        fig, ax = plt.subplots(1, 1, figsize=(10,6))

        # Make it pretty/consistent
        define_axis_style(ax = ax, title=f'Top {n} Features: {target_names[cat]}', y_label=None, x_label="Log Probability")
        plot_feature_importances(
                                ax = ax,
                                feat_importances = nb_model.feature_log_prob_[cat],
                                feat_std_deviations = [],
                                feat_names = feature_words,
                                n_features = n,
                                outfilename = f'images/nb_top_features_12_{target_names[cat]}'
                            )
        
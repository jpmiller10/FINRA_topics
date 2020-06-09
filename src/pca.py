import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from clean import to_categorical, to_numeric
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, animation


plt.style.use('ggplot')
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}

def plot_pca_three_components(X, y, title='PCA_3components'):
    '''Function for creating a 3D gif showing the
    frist three principal componets
    X: numpy.array, shape (n, 300)
    A two dimensional array containing the vectorized claims
    y: numpy.array
    The labels of the datapoints.  1 or 0.
    title: str
    A title for the plot.
    '''

    n_components = 3
    pca = PCA(n_components=n_components) #pca object
    X_pca = pca.fit_transform(X)

    xx = X_pca[:, 0]
    yy = X_pca[:, 1]
    zz = X_pca[:, 2]

    #Creating arays for axes & axes limits
    zeros = np.zeros(100)
    xx_center= np.mean(X_pca[:, 0])
    xx_range = np.max(X_pca[:, 0]) - np.min(X_pca[:, 0])*.5
    yy_center= np.mean(X_pca[:, 1])
    yy_range = np.max(X_pca[:, 1]) - np.min(X_pca[:, 1])*.5
    zz_center= np.mean(X_pca[:, 2])
    zz_range = np.max(X_pca[:, 2]) - np.min(X_pca[:, 2])*.5
    x_axis = np.linspace(xx_center-xx_range, xx_center+xx_range, 100)
    y_axis = np.linspace(yy_center-yy_range, yy_center+yy_range, 100)
    z_axis = np.linspace(zz_center-zz_range, zz_center+zz_range, 100)

    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)

    def init():
        # Plot the surface.
        ax.scatter(xx, yy, zz, marker='o', cmap=cm.coolwarm, s=10, c=y, alpha=0.3)
        ax.plot(x_axis,zeros,zeros, linewidth=0.5, color='k', alpha =1)
        ax.plot(zeros, y_axis,zeros, linewidth=0.5, color='k', alpha =1)
        ax.plot(zeros, zeros, z_axis, linewidth=0.5, color='k', alpha =1)
        ax.set_title(title)
        ax.axis('off')
        ax.set_xlim(xx_center-xx_range/2, xx_center+xx_range/2)
        ax.set_ylim(yy_center-yy_range/2, yy_center+yy_range/2)
        ax.set_zlim(zz_center-zz_range/2, zz_center+zz_range/2)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return fig,

    def animate(i):
        # azimuth angle : 0 deg to 360 deg
        ax.view_init(elev=195, azim=i*2)
        return fig,

    # Animate
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=180, interval=180, blit=True)
    fn = title
    ani.save('images/'+fn+'.gif',writer='PillowWriter',fps=5,savefig_kwargs={'transparent': True, 'facecolor': 'none'})

if __name__ == "__main__":
    
    df_finra = pd.read_pickle('data/finra_pickled_df_top')
    df_finra['targets_1'] = df_finra['targets_1'].replace('Unfavorable', 0)
    df_finra['targets_1'] = df_finra['targets_1'].replace('Favorable', 1)
    df_finra['targets_2'] = df_finra['targets_2'].replace('Unfavorable', 0)
    df_finra['targets_2'] = df_finra['targets_2'].replace('Settled', 1)
    df_finra['targets_2'] = df_finra['targets_2'].replace('Favorable', 2)
    to_numeric(df_finra, ['targets_1', 'targets_2'])
    
    X = df_finra['allegations']
    y = df_finra['targets_1']
    y2 = df_finra['targets_2']
    y3 = df_finra['topics']
    
    count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_df=1.0, min_df=1, max_features=None)

    count_vect.fit(X)

    X_counts = count_vect.transform(X)
    

    tfidf_transformer = TfidfTransformer(use_idf=True)
    tfidf_transformer.fit(X_counts)
    X_tfidf = tfidf_transformer.transform(X_counts)
    X_tfidf = X_tfidf.todense()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tfidf)
    evr = pca.explained_variance_ratio_
    print(evr)
    print("The 2 principal components explain {0:0.1f}%"
        " of the variance in the original data.".format(evr.sum()*100))

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
    #         cmap=cm.coolwarm, edgecolor='k', s=40)
    # ax.set_title("First Two PCA Directions with Two Targets")
    # ax.set_xlabel("1st eigenvector (PC1)")
    # ax.set_ylabel("2nd eigenvector (PC2)");
    # plt.savefig("images/PCA_2com_cool.png")


    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y2, 
    #         cmap=cm.coolwarm, edgecolor='k', s=40)
    # ax.set_title("First Two PCA Directions with Three Targets")
    # ax.set_xlabel("1st eigenvector (PC1)")
    # ax.set_ylabel("2nd eigenvector (PC2)");
    # plt.savefig("images/PCA_3com_cool.png")


    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y3, 
    #         cmap=cm.coolwarm, edgecolor='k', s=40)
    # ax.set_title("First Two PCA Directions by Topic")
    # ax.set_xlabel("1st eigenvector (PC1)")
    # ax.set_ylabel("2nd eigenvector (PC2)");
    # plt.savefig("images/PCA_topics_cool.png")

    plot_pca_three_components(X_tfidf, y, title='PCA_3comp_2targets_veryslow')
    # plot_pca_three_components(X_tfidf, y2, title='PCA_3comp_3targets')
    plot_pca_three_components(X_tfidf, y3, title='PCA_3comp_Topics_veryslow')
    

    # pca = PCA(n_components=3)
    # X_pca = pca.fit_transform(X_tfidf) 
    # evr = pca.explained_variance_ratio_
    # print(evr)
    # print("The 3 principal components explain {0:0.1f}%"
    #     " of the variance in the original data.".format(evr.sum()*100))

    # fig = plt.figure() 
    # ax = fig.add_subplot(111, projection='3d') 
    # ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, 
    #             cmap=plt.cm.Set1, edgecolor='k', s=40) 
    # ax.set_xlabel('PC1') 
    # ax.set_ylabel('PC2') 
    # ax.set_zlabel('PC3') 
    # plt.savefig(f"images/PCA_3d.png")
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from features import clean_column, get_stop_words
from string import punctuation
import matplotlib.pyplot as plt
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

def make_wordcloud(arr1, arr2, title, stop_words):
    final_text = ""
    for topic in np.sort(arr1.unique()):
        final_text = ""
        for row in zip(arr2, arr1):
            if row[1] == topic:
                row_text = str(row[0]).replace("\n", "")
                final_text += " ".join([word for word in row_text.split(" ") if word not in stop_words])
        wordcloud = WordCloud(width = 800, height = 800, background_color ='black', 
                        stopwords=stop_words).generate(str(final_text))
        plt.title(f"{title} {topic}")
        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"images/wordcloud_{title}{topic}.png")
    

if __name__ == "__main__":
    df_finra = pd.read_pickle('data/finra_pickled_df_top')

    clean_column(df_finra, 'allegations', punctuation)

    additional_stop_words = [
        "the",
        "raymond",
        "james",
        "alleged",
        "involvedaccount",
        "involveduse"
    ]
    stop_words = get_stop_words(additional_stop_words)

    # make_wordcloud(df_finra['topics'], df_finra['allegations'], 'Topic', stop_words)
    # make_wordcloud(df_finra['targets_1'], df_finra['allegations'], '2_Targets', stop_words)
    make_wordcloud(df_finra['targets_2'], df_finra['allegations'], '3_Targets', stop_words)
   
    # all_allegations = df_finra['allegations'].str.cat(sep=' ')
    # wordcloud = WordCloud(width = 800, height = 800, 
    #             background_color ='white', 
    #             stopwords = stop_words.union({'great'}), 
    #             min_font_size = 10, max_words=200, scale=3).generate(all_allegations)
    # plt.figure(figsize = (5, 5), facecolor = None) 
    # plt.imshow(wordcloud) 
    # plt.axis("off") 
    # plt.tight_layout(pad = 0) 
    # plt.savefig('images/wordcloud_all_allegations.png', dpi=80)
    # final_text = ""
    # for topic in np.sort(df_finra['topics'].unique()):
    #     final_text = ""
    #     for row in zip(df_finra['allegations'], df_finra['topics']):
    #         if row[1] == topic:
    #             row_text = str(row[0]).replace("\n", "")
    #             final_text += " ".join([word for word in row_text.split(" ") if word not in stop_words])
    #     wordcloud = WordCloud(stopwords=stop_words).generate(str(final_text))
    #     plt.title(f"Topic #{topic}")
    #     # Display the generated image:
    #     plt.imshow(wordcloud, interpolation='bilinear')
    #     plt.axis("off")
    #     plt.tight_layout()
    #     plt.savefig(f"images/wordcloud_topic{topic}.png")
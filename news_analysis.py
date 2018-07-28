#text clustering LDA
#text processing
#visualizations

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import lda
import logging
logging.getLogger("lda").setLevel(logging.WARNING)
from sklearn.manifold import TSNE
import numpy as np
import bokeh.plotting as bp
from bokeh.io import output_notebook
from bokeh.resources import INLINE
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_file

news = pd.read_csv("news.csv")
#print(news.head())

news = news.drop_duplicates("description")
news = news[~news["description"].isnull()]
news[~news["description"].apply(lambda x: len(x.split(" ")) < 10)]   #Dropping articles with description less than 10 words
news.reset_index(inplace=True, drop=True)        #Reset index

print(news.shape)

#PLotting distribution of news description lengths
plt.xlabel("Length")
plt.ylabel("Number of News Descriptions")
plt.title("Distribution of Description Lengths")
plt.show(news.description.map(len).hist(figsize = (15, 5), bins = 100))

#Removing stop words. Tokenizing. Calculating each token count, retaining those with count >= 5. Calculating TfIDF scores
count_vect = CountVectorizer(min_df=5, analyzer='word', stop_words = "english", ngram_range = (1, 2))
news_token_matrix = count_vect.fit_transform(news["description"])
tfidf_transformer = TfidfTransformer()
news_tfidf_matrix = tfidf_transformer.fit_transform(news_token_matrix)

#Plotting distribution of TfIdf scores
tfidf = dict(zip(count_vect.get_feature_names(), tfidf_transformer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
tfidf.columns = ['tfidf']
plt.xlabel("TfIDF Scores")
plt.ylabel("Number of News Descriptions")
plt.title("Distribution of TfIDF Scores")
plt.show(tfidf.tfidf.hist(bins=25, figsize=(15,7)))

#Creating word cloud
def plot_word_cloud(terms, category):
    text = terms.index
    text = ' '.join(list(text))
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure(figsize=(25, 25))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Words with {} TfIdf Scores.".format(category))
    plt.show()

#Lowest TfIDF scores
plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=True).head(40), "Lowest")
#Highest TfIDF scores
plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=False).head(40), "Highest")

#Kmeans separates the documents into disjoint clusters. the assumption is that each cluster is attributed a single topic.
#However, descriptions may in reality be characterized by a "mixture" of topics.
#For example, let's take an article that deals with the hearing that Zuckerberg had in front of the congress:
#you'll obviously have different topics rising based on the keywords: Privacy, Technology, Facebook app, data, etc.
#We'll cover how to deal with this problem with the LDA algorithm.

n_topics = 10
n_iter = 2000
lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
X_topics = lda_model.fit_transform(news_token_matrix)

n_top_words = 20
topic_summaries = []

topic_word = lda_model.topic_word_  # get the topic words
vocab = count_vect.get_feature_names()
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

#dimensionality reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_lda = tsne_model.fit_transform(X_topics)

doc_topic = lda_model.doc_topic_
lda_keys = []
for i, tweet in enumerate(news['description']):
    lda_keys += [doc_topic[i].argmax()]


colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
"#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])

plot_lda = bp.figure(plot_width=700, plot_height=600, title="LDA topic visualization",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

lda_df = pd.DataFrame(tsne_lda, columns=['x','y'])
lda_df['description'] = news['description']
lda_df['category'] = news['category']
lda_df['topic'] = lda_keys
lda_df['topic'] = lda_df['topic'].map(int)
lda_df["colors"] = colormap[lda_keys]
plot_lda.scatter(source=lda_df, x='x', y='y', color= "colors")

hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips={"description":"@description", "topic":"@topic", "category":"@category"}
show(plot_lda)

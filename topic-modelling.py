import numpy as np 
import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import string
from sklearn.decomposition import NMF, LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
import gensim

nlp = spacy.load('en_core_web_sm') 

data = pd.read_csv("feedbackdata.csv")
data = data.filter(['feedback_comment'])
data = data.dropna()
data = data.drop_duplicates()

def detect_non_english_words(data):
    for index, rows in data.iterrows():
        feedbackComment = rows['feedback_comment']
        badWordCount = 0
        splitComment = feedbackComment.split()
        feedbackCommentLength = len(splitComment)
        for word in splitComment:
                if not wordnet.synsets(word):
                  #  print(word)
                    badWordCount +=1
        if(badWordCount > 0 and badWordCount/feedbackCommentLength >= 0.5):
            data.drop(index, inplace=True)
    return data

data = detect_non_english_words(data)
comments = []
for i in data['feedback_comment']:
    comments.append(str(i))

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

comments_clean = [clean(comment).split() for comment in comments]

from gensim import corpora
dictionary = corpora.Dictionary(comments_clean)
document_term_matrix = [dictionary.doc2bow(word) for word in comments_clean]
Lda = gensim.models.ldamodel.LdaModel

ldamodel = Lda(document_term_matrix, num_topics=3, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=5, num_words=3))


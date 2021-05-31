import numpy as np 
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

#Read into Dataframe
data = pd.read_csv("feedbackCommentsOnly.csv")
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

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

feedbackComments = data['feedback_comment']

import json
sentimentValue = []
sentiment = []
comment = []

sentimentDF = pd.DataFrame(columns = ('comment', 'sentiment', 'sentimentValue'))
for i in feedbackComments:
    i = str(i)
    comment.append(str(i))
    result = nlp.annotate(str(i), properties={
            'annotators': 'sentiment', 
            'timeout': 100000,
        })
    result = json.loads(result)
    
    for k in result['sentences']:
        sentiment.append(str(k['sentiment']))
        sentimentValue.append(str(k['sentimentValue']))
        row = [i, k['sentiment'], k['sentimentValue']]
        sentimentDF.loc[len(sentimentDF)] = row
                         
sentimentDF.drop_duplicates('comment')
sentimentDF.to_csv('sentimentDF.csv', index=False)

positive = sentimentDF[sentimentDF['sentiment'] == 'Positive']
positive = positive.drop_duplicates()
pd.set_option('display.max_colwidth', -1)

neutral = sentimentDF[sentimentDF['sentiment'] == 'Neutral']
negative = sentimentDF[sentimentDF['sentiment'] == 'Negative']

blobComment = []
blobPolarity = []
blobSubjectivity = []

for i in range(0, len(feedbackComments)):
    blob = TextBlob(feedbackComments.iloc[i])
    blobComment.append(str(blob))
    blobPolarity.append(str(blob.sentiment.polarity))
    blobSubjectivity.append(str(blob.sentiment.subjectivity))
    
blobDF = pd.DataFrame({
        'Comment': blobComment,
        'Polarity': blobPolarity,
        'Subjectivity': blobSubjectivity
        })
blobDF = blobDF[['Comment', 'Polarity', 'Subjectivity']]
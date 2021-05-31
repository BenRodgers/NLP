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

nlp = spacy.load('en_core_web_sm') 

data = pd.read_csv("feedbackdata.csv")
data = data.filter(['feedback_comment'])
data = data.dropna()
data = data.drop_duplicates() 
def to_lower(data):
    for i in range(0, len(data)):
        data.iloc[i] = data.iloc[i].str.lower()
      
    return data

def remove_punctuation(data):
    data['feedback_comment'] = data['feedback_comment'].str.replace('[{}]'.format(string.punctuation), '')
           
    return data
def remove_stop_words(data):
    stop = stopwords.words('english')
    #Loop through each comment
    for i in range(0, len(data)):
        modifiedComment = []
        feedbackComment = data.iloc[i]['feedback_comment']
        splitComment = feedbackComment.split()
        for word in splitComment:  
            if (word) not in stop:
                modifiedComment.append(word)
                modifiedComment.append(' ')
                
        newComment = ''.join(modifiedComment)
        data.iloc[i]['feedback_comment'] = newComment
    return data

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
            
def preprocess(data):
    data = to_lower(data)
    data = remove_punctuation(data)
    data = remove_stop_words(data)
    data = detect_non_english_words(data)
    
    return data

data = preprocess(data)  
#Calculate TF-IDF
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['feedback_comment'])
feature_names = vectorizer.get_feature_names()

Adjective = []
words = " "
doc = words.join(feature_names)
doc = nlp(doc) 

for token in doc:
    if token.dep_ == 'amod' and token.pos_ == 'ADJ':
        Adjective.append(str(token))

for a in Adjective:
    token = nltk.word_tokenize(a)
    print(token)
    if not wordnet.synsets(a) or nltk.pos_tag(token) != 'JJ':
        
        Adjective.remove(a) 
        
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns = feature_names)

df = df[df.columns.intersection(Adjective)]

topTwentyWords = dict(df.max().nlargest(10))

for word in topTwentyWords:
    analyser = SentimentIntensityAnalyzer()
 #   print(analyser.polarity_scores(word)['compound'])
    

topTwentyDF = pd.DataFrame()
topTwentyDF['words'] = topTwentyWords.keys()

print(topTwentyDF)
polarity = []
for i in topTwentyDF['words']:
    blob = TextBlob(i)
    polarity.append((i, blob.polarity))
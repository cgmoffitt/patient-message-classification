""" Import Packages and Data """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
from time import time
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

df = pd.read_csv('./data/patient-message-data-1.csv', header=0,index_col=0)
print("------------------------------------------")
print("Initial Look at Data: ")
print(df.head())

print("Data Shape: ", df.shape)
print("Initial # of Words: ", df['msg_txt'].apply(lambda x: len(x.split(' '))).sum())



""" Clean Data """

#checking missing values
print("\n------------------------------------------")
print("Checking missing values")
print(df.isnull().sum())


#changing data type to string
df['msg_txt'] = df['msg_txt'].astype(str)

#Remove respondent types with less than 9,000 associated data
print("\n------------------------------------------")
print("Initial respondent_types pre pruning:")
print(df.respondent_type.value_counts())
respondent_types_to_remove = []
for val, cnt in df.respondent_type.value_counts().iteritems():
    if cnt < 9000:
        respondent_types_to_remove.append(val)
df = df[df.respondent_type.isin(respondent_types_to_remove) == False]

print("\n------------------------------------------")
print("Respondent types post proning:\n")
print(df.respondent_type.value_counts())
print("New data shape: ", df.shape)

#Remove unnecessary text
stop = stopwords.words('english')
lem = WordNetLemmatizer()

def cleanText(words):
    """The function to clean text"""
    words = re.sub("[^a-zA-Z]"," ",words)
    text = words.lower().split()
    return " ".join(text)
df['msg_txt'] = df['msg_txt'].apply(cleanText)

def remove_stopwords(text):
    """The function to removing stopwords"""
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

def word_lem(text):
    """The function to apply lemmatizing"""
    lem_text = [lem.lemmatize(word) for word in text.split()]
    return " ".join(lem_text)

df['msg_txt'] = df['msg_txt'].apply(cleanText)
df['msg_txt'] = df['msg_txt'].apply(remove_stopwords)
df['msg_txt'] = df['msg_txt'].apply(word_lem)

#New Head
print("New Data Post Cleaning:\n")
print(df.head())

df['msg_txt'] = df['msg_txt'].astype(str)
print("\nNew total words:\n")
print(df['msg_txt'].apply(lambda x: len(x.split(' '))).sum())

#save clean data
df.to_csv('./data/patient-messages-clean.csv', encoding='utf-8')
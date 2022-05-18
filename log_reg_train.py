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
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

""" Load Data """
df = pd.read_csv('./data/patient-messages-clean.csv', header=0,index_col=0)
print("------------------------------------------")
print("Initial Look at Data: ")
print(df.head())

print("Data Shape: ", df.shape)
print(df.respondent_type.value_counts())




""" Data preparation """
print("\n------------------------------------------")
print("Data preparation:")
df['msg_txt'] = df['msg_txt'].astype(str)
train, test = train_test_split(df, test_size=0.3, random_state=42)

print("\nLook at train Data:")
print(train.respondent_type.value_counts())

print("\nLook at test Data:")
print(test.respondent_type.value_counts())

print("\nCreating tagged documents...")
train_tag = train.apply(lambda x: TaggedDocument(words=word_tokenize(x['msg_txt']), tags=[x.respondent_type]), axis=1)
test_tag = test.apply(lambda x: TaggedDocument(words=word_tokenize(x['msg_txt']), tags=[x.respondent_type]), axis=1)

print("\nGlance at Train example: ")
print(train_tag.iloc[0])

print("\nGlance at Test example: ")
print(test_tag.iloc[0])







""" Build Model """
print("\n------------------------------------------")
print("Building Model...")
doc_model = Doc2Vec(dm=0, vector_size=100, min_count=2, window=2, sample=0)
doc_model.build_vocab(train_tag)
print("Training vocabulary", doc_model.corpus_total_words)

print("\nTraining Doc2Vec model...")
doc_model.train(train_tag, total_examples=doc_model.corpus_count, epochs=30)

#save model
doc_model.save('model.doc2vec')

def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, feature_vectors

y_train, X_train = vector_for_learning(doc_model, train_tag)
y_test, X_test = vector_for_learning(doc_model, test_tag)








""" Train Logistic Regression """
print("\n------------------------------------------")
print("Training Logistic Regression Model...")
log_reg = LogisticRegression(n_jobs=1, C=5)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)







""" Testing Model """
print("\n------------------------------------------")
print("Testing Model...")
print('Testing accuracy %s' % accuracy_score(y_pred, y_test))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
ytest = np.array(y_test)
print(classification_report(ytest, y_pred))
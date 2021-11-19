# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:33:48 2021

@author: cmbbd
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import re
import sklearn as sk
import html
from sklearn.metrics import confusion_matrix , classification_report
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sub_df = pd.read_csv('sample_submission.csv')

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)

def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)

import string


def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)



def prepro(train):
    train["text"] = train.text.map(lambda x: remove_URL(x))
    train["text"] = train.text.map(lambda x: remove_html(x))
    train["text"] = train.text.map(lambda x: remove_emoji(x))
    train["text"] = train.text.map(lambda x: remove_punct(x))

prepro(train)



X = train
y = train.target




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv.fit(X['text'])


def features(X,cv):
    inputs_cv= cv.transform(X['text'])
    return(inputs_cv)



inputs_cv = features(X,cv)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test1=train_test_split(inputs_cv,y,test_size=0.2,random_state=5)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(x_train,y_train)
target_pred = naive_bayes.predict(x_test)
Report = classification_report(y_test1,target_pred)

from sklearn.svm import SVC
model = SVC(probability=True)
model.fit(x_train, y_train)
ypred = model.predict(x_test)
Report1 = classification_report(y_test1,ypred)


MODEL_pred = model.predict_proba(x_test)
CLF_pred = naive_bayes.predict_proba(x_test)
ALPHA = [i/10000 for i in range(10000)]
LOSS = [sk.metrics.log_loss(y_test,(alpha*CLF_pred+(1-alpha)*MODEL_pred)) for alpha in ALPHA]
alpha = ALPHA[np.argmin(LOSS)]

e = sklearn.ensemble.VotingClassifier([('1',clf),('2',model)],voting = 'soft')
e = VotingClassifier([('1',clf),('2',model)],voting = 'soft')
e = VotingClassifier([('1',naive_bayes),('2',model)],voting = 'soft')
e.fit(X_train, y_train)
e.fit(x_train, y_train)
e.score(x_test,y_test1)

Report2 = classification_report(y_test1,np.argmax(np.argmax(alpha*CLF_pred)+(1-alpha)*MODEL_pred,axis=1))
Report3 = classification_report(y_test1,e.predict(x_test))
TEST = prepro(test)
TEST_padded = features(test)
TEST_cv  = features(test,cv)



MODEL_pred = model.predict_proba(TEST_padded)
CLF_pred = naive_bayes.predict_proba(TEST_cv)
predictions = [[False, True].index(i) for i in 0.5<((alpha*CLF_pred[:,1]).reshape(len(CLF_pred[:,1]),1)+(1-alpha)*MODEL_pred)]

sub_df['target'] = predictions
sub_df.to_csv("submission.csv", index=False)
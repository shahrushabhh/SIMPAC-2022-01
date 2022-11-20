#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
import re
import string
from wordcloud import STOPWORDS
from sklearn.metrics import accuracy_score, precision_score
import PySimpleGUI as sg
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r'C:\Users\Rushabh Shah\Downloads\cyber-operations-incidents.csv')
y=df["Type"]
ID_X=df["Description"]
ID_X=ID_X.astype(str)
def text_clean_1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text

cleaned1 = lambda x: text_clean_1(x)
ID_X=ID_X.apply(cleaned1)

def text_clean_2(text):
    text= re.sub('[''""...]','',text)
    text=re.sub('\n','',text)
    return text

cleaned2 = lambda x: text_clean_2(x)
ID_X=ID_X.apply(cleaned2)

def text_clean_3(text):
    text= re.sub('_',' ',text)
    return text

cleaned3 = lambda x: text_clean_3(x)
ID_X=ID_X.apply(cleaned3)

ID_X_train, ID_X_test, y_train, y_test = train_test_split(ID_X, y,test_size=0.25, random_state=20)

stopwords = set(STOPWORDS)

ID_text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators=100)),
                     ])

ID_text_clf.fit(ID_X_train,y_train)
labels=ID_text_clf.predict(ID_X_test)


def classify_category(text):
    text=ID_text_clf.predict([text])
    return text


sg.change_look_and_feel('GreenTan')

layout = [ [sg.Text('Enter cyber incident details to classify into category'), sg.Multiline(size=(31,8),key='-CID-', do_not_clear=True)],
           [sg.Text('category = '),
            sg.Text(size=(20,1), key='-OUT-CIG-')],
           [sg.Button('Classify the category', bind_return_key=True), sg.Button('Quit')]  ]

window = sg.Window("Cyber securtiy incident classifier", layout)
# Loop, reading events (button clicks) and getting input field
while True:             # Event Loop
    event, values = window.read()
    if event in (None, 'Quit'):
        break
    if event == 'Classify the category':
          a=classify_category(values['-CID-'])
          window['-OUT-CIG-'].Update(a)       
window.close()


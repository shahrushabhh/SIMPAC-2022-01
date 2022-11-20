#!/usr/bin/env python
# coding: utf-8

# # Cyber Security Data Analysis

# # Importing necessary libraries

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


# # Loading Data and Preprocessing

# In[2]:


df = pd.read_csv(r'C:\Users\Rushabh Shah\Downloads\cyber-operations-incidents.csv')


# In[3]:


df.tail(5)


# In[4]:


y=df["Type"]
ID_X=df["Description"]


# In[5]:


type(ID_X)


# # Data Cleaning

# In[6]:


ID_X=ID_X.astype(str)


# In[7]:


import re
import string


# In[8]:


def text_clean_1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text

cleaned1 = lambda x: text_clean_1(x)


# In[9]:


ID_X=ID_X.apply(cleaned1)


# In[10]:


ID_X


# In[11]:


def text_clean_2(text):
    text= re.sub('[''""...]','',text)
    text=re.sub('\n','',text)
    return text

cleaned2 = lambda x: text_clean_2(x)


# In[12]:


ID_X=ID_X.apply(cleaned2)


# In[13]:


def text_clean_3(text):
    text= re.sub('_',' ',text)
    return text

cleaned3 = lambda x: text_clean_3(x)


# In[14]:



ID_X=ID_X.apply(cleaned3)


# # Incident category prediction Model building from Incident detail

# In[15]:


ID_X_train, ID_X_test, y_train, y_test = train_test_split(ID_X, y,test_size=0.25, random_state=20)


# In[16]:


from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)


# In[17]:


ID_text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators=100)),
                     ])


# In[18]:


ID_text_clf.fit(ID_X_train,y_train)


# In[19]:


labels=ID_text_clf.predict(ID_X_test)
labels


# In[20]:


type(y_train)


# In[21]:


y_test=pd.DataFrame(y_test)


# In[22]:


y_test


# In[23]:


from sklearn.metrics import accuracy_score, precision_score

print("Accuracy: ", round(accuracy_score(labels, y_test)*100,2) , "%")
print("Precision: ", round(precision_score(labels,y_test, average = "weighted")*100,2) , "%")


# In[24]:


example=["Hidden Cobra used a variety of malware tools to hack into and steal money from banks, cryptocurrency exchanges, and ATMs."]
level=ID_text_clf.predict(example)
level


# In[25]:


def classify_category(text):
    
    text=ID_text_clf.predict(text)
    return text


# In[26]:


a=classify_category(["Hidden Cobra used a variety of malware tools to hack into and steal money from banks, cryptocurrency exchanges, and ATMs."])


# In[27]:


a


# # Model Prediction using Naive Bayes

# In[ ]:


nb_model = make_pipeline (TfidfVectorizer(),MultinomialNB())


# In[ ]:


nb_model.fit(ID_X_train,y_train)


# In[ ]:


labels= nb_model.predict(ID_X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score

print("Accuracy: ", round(accuracy_score(labels, y_test)*100,2) , "%")
print("Precision: ", round(precision_score(labels,y_test, average = "weighted")*100,2) , "%")


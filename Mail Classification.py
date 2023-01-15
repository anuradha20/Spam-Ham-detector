#!/usr/bin/env python
# coding: utf-8

# ## Email Spam/Ham Detector

# In[103]:


import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer


# In[92]:


df=pd.read_csv("C:\\Users\\Dell\\Downloads\\spam.csv")
df.head()


# In[93]:


def num(x):
    if(x=="ham"):
        return 1
    else:
        return 0


# In[94]:


df["Category"] = df.Category.apply(lambda x:num(x))
df.head()


# In[95]:


X=df.Message


# In[96]:


y=df.Category


# In[97]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)


# In[104]:


clf = Pipeline([
    ("vectorizer" , CountVectorizer()),
    ("model", MultinomialNB())
])


# In[105]:


clf.fit(X_train,y_train)


# In[106]:


clf.score(X_test,y_test)


# In[107]:


print(classification_report(y_test,clf.predict(X_test)))


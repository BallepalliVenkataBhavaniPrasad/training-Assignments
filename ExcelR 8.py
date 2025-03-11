#!/usr/bin/env python
# coding: utf-8

# In[1]:


Text = "I am learning NLP"


# In[2]:


import pandas as pd
pd.get_dummies(Text.split())


# In[3]:


text=["i love EDITING and i will learn EDITING in 2month"]


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
vectorizer.fit(text)
vector=vectorizer.transform(text)


# In[6]:


print(vectorizer.vocabulary_)
print(vector.toarray())


# In[7]:


print(vector)


# In[8]:


get_ipython().run_line_magic('pinfo', 'CountVectorizer')


# In[10]:


df=pd.DataFrame(data=vector.toarray(),columns=vectorizer.get_feature_names_out())
df


# In[11]:


text='I am learning EDITING'


# In[13]:


import nltk
nltk.download('punkt_tab')
  


# In[14]:


from textblob import TextBlob
TextBlob(text).ngrams(1)


# In[16]:


TextBlob(text).ngrams(2)


# In[17]:


TextBlob(text).ngrams(3)


# In[19]:


TextBlob(text).ngrams(4)


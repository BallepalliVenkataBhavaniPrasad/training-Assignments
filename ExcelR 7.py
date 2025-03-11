#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset=pd.read_csv(r"C:\Users\prasa\OneDrive\Documents\hate_speech.csv")
dataset.head()


# In[2]:


dataset.shape


# In[3]:


dataset.label.value_counts()


# In[4]:


for index, tweet in enumerate(dataset["tweet"][10:15]):
    print(index+1,"-",tweet)


# In[5]:


import re
def clean_text(text):
    text=re.sub(r'[^a-zA-Z\']',' ', text)
    text=re.sub(r'[^\x00-\x7F]+',' ', text)
    text=text.lower()
    return text


# In[6]:


dataset['clean_text']=dataset.tweet.apply(lambda x: clean_text(x))


# In[7]:


from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[ ]:





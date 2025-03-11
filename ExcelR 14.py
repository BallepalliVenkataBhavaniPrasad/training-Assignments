#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\prasa\Downloads\emotion.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df.label.value_counts()


# In[7]:


import seaborn as sns
sns.countplot(x=df.label)


# In[8]:


df['text']=df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[9]:


import nltk
nltk.download('stopwords')


# In[10]:


from nltk.corpus import stopwords
stop=stopwords.words('english')
df['text']=df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[11]:


get_ipython().system('pip install textblob')


# In[12]:


from nltk.stem import WordNetLemmatizer
from textblob import Word
df['text']=df['text'].apply(lambda X: " ".join([Word(word).lemmatize() for word in X.split()]))
df['text'].head()


# In[13]:


X=df[['text']]
y=df['label']


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(X,y, test_size=0.2)


# In[15]:


print(x_train.shape, y_train.shape)


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
tfidf.fit(df['text'])
xtrain_tfidf=tfidf.transform(x_train['text'])
xtest_tfidf=tfidf.transform(x_test['text'])


# In[18]:


from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
pclf=PassiveAggressiveClassifier()
pclf.fit(xtrain_tfidf, y_train)
pred=pclf.predict(xtest_tfidf)
print(metrics.classification_report(y_test, pred))


# In[ ]:





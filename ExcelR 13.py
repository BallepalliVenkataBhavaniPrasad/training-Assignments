#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv(r"C:\Users\prasa\Downloads\fake_news.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isna().sum()


# In[6]:


data=data.drop(['id'], axis=1)


# In[7]:


data=data.fillna('')


# In[8]:


data['content']=data['author']+' '+ data['title']+' '+data['text']


# In[9]:


data=data.drop(['title','author','text'],axis=1)


# In[10]:


data.head()


# In[11]:


data['content']=data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[12]:


data['content']=data['content'].str.replace('[^\w\s\]','')


# In[13]:


import nltk
nltk.download('stopwords')


# In[14]:


from nltk.corpus import stopwords
stop=stopwords.words('english')
data['content']=data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[15]:


get_ipython().system('pip install textblob')


# In[16]:


from nltk.stem import WordNetLemmatizer
from textblob import Word
data['content']=data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['content'].head()


# In[17]:


X=data[['content']]
y=data['label']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=45, stratify=y)


# In[20]:


print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (X_test.shape)


# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(data['content'])
xtrain_tfidf = tfidf_vect.transform(X_train['content'])
xtest_tfidf = tfidf_vect.transform(X_test['content'])


# In[23]:


from sklearn import model_selection, linear_model, metrics


# In[24]:


from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
pclf=PassiveAggressiveClassifier()
pclf.fit(xtrain_tfidf, y_train)
predictions=pclf.predict(xtest_tfidf)
print(metrics.classification_report(y_test, predictions))


# In[25]:


print(metrics.confusion_matrix(y_test,predictions))


# In[26]:


from sklearn.neural_network import MLPClassifier
mlpclf = MLPClassifier(hidden_layer_sizes=(256,64,16),
                       activation='relu',
                       solver='adam')
mlpclf.fit(xtrain_tfidf, y_train)
predictions = mlpclf.predict(xtest_tfidf)
print(metrics.classification_report(y_test, predictions))


# In[27]:


print(metrics.confusion_matrix(y_test,predictions))


# In[28]:


import pickle
pickle.dump(pclf, open("fakenews1.pkl", 'wb'))


# In[29]:


loaded_model = pickle.load(open("fakenews1.pkl", 'rb'))
result = loaded_model.score(X_test.values, y_test.values)
print(result)


# In[ ]:


def fake_news_det(news):
    input_data=[news]
    vectorized_input_data=tfidf_vect.transform(input_data)
    prediction=pclf.predict(vectorized_input_data)
    print(prediction)


# In[ ]:


fake_news_det('U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.')


# In[ ]:





# In[ ]:





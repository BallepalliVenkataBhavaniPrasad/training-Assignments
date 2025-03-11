#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Reviews.csv',nrows = 500)
df.head()


# In[2]:


df.info()


# In[3]:


df.Summary.head()


# In[4]:


df.Text.head()


# In[5]:


get_ipython().system('pip install nltk')
import nltk
nltk.download('stopwords')


# In[6]:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
#Lowercasing
df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['Text'] = df['Text'].str.replace('[^\w\s]','')
#stopwords Removal
stop = stopwords.words('english')
df['Text'] = df['Text'].apply(lambda x:" ".join(x for x in x.split() if x not in stop))
#Spelling correction
df['Text'] = df['Text'].apply(lambda x: str(TextBlob(x).correct()))
#Lemmatization
df['Text']=df['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df.Text.head()


# In[ ]:


df.Text.head()


# In[ ]:


reviews=df
reviews.dropna(inplace=True)
reviews.Score.hist(bins=5,grid=False)
plt.show()
print(reviews.groupby('Score').count().Id)


# In[ ]:


score_1=reviews[reviews['Score']==1].sample(n=18)
score_2=reviews[reviews['Score']==2].sample(n=18)
score_3=reviews[reviews['Score']==3].sample(n=18)
score_4=reviews[reviews['Score']==4].sample(n=18)
score_5=reviews[reviews['Score']==5].sample(n=18)


# In[ ]:


review_sample = pd.concat([score_1,score_2,score_3,score_4,score_5],axis=0)
review_sample.reset_index(drop=True,inplace=True)
print(review_sample.groupby('Score').count().Id)


# In[ ]:


from wordcloud import WordCloud
reviews_str=" ".join(reviews_sample["Summary"].to_numpy())
wordcloud=WordCloud(background_color='white').generate(reviews_str)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


negative_reviews=reviews_sample[reviews_sample['Score'].isin([1,2]) ]
positive_reviews=reviews_sample[reviews_sample['Score'].isin([4,5]) ]
negative_reviews_str=negative_reviews.Summary.str.cat()
positiive_reviews_str=positive_reviews.Summary.str.cat()


# In[ ]:


wordcloud_negative = WordCloud(background_color='white').generate(negative_reviews_str)
wordcloud_positive = WordCloud(background_color= 'white').generate(positive_reviews_str)
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.imshow(wordcloud_negative, interpolation='bilinear')
ax1.axis("off")
ax1.set_title('Reviews with Negative Scores', fontsize=20)
ax2 = fig.add_subplot(212)
ax2.imshow(wordcloud_positive, interpolation='bilinear')
ax2.axis("off")
ax2.set_title('Reviews with Positive Scores', fontsize=20)
plt.show()


# In[ ]:


get_ipython().system('pip install vaderSentiment')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
plt.style.use('fivethirtyeight')
cp = sns.color_palette()
analyzer = SentimentIntensityAnalyzer()


# In[ ]:


df_c = pd.concat([df.reset_index(drop=True), df_sentiments], axis=1)
df_c.head(3)


# In[ ]:


df_c['Sentiment']=np.where(df_c['compound'] >=0 , 'Positive', ['Negative'])


# In[ ]:





# In[ ]:





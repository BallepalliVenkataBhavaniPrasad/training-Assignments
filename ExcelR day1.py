#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
dataset = pd.read_csv('tweets.csv', encoding='ISO-8859-1')
print(dataset.head(3))  


# In[11]:


def gen_freq(text):
    word_list = []

    for tw_words in text.split():
        word_list.extend(tw_words)

    word_freq = pd.Series(word_list).value_counts()

    word_freq[:10]
    
    return word_freq

word_freq = gen_freq(dataset.text.str)
word_freq


# In[ ]:


word_freq = gen_freq(dataset.text.str)


# In[12]:


get_ipython().system('pip install wordcloud')


# In[15]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(12, 8))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[1]:


import pandas as pd
text=['Sarah lives in a hut in the village.',
     'She has an apple tree in her backyard.',
     'The apples are not in red colour.']
df = pd.DataFrame(text, columns=['Sentence'])
df


# In[3]:


get_ipython().system('pip install spacy')


# In[4]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[8]:


import spacy
nlp=spacy.load("en_core_web_sm")
token=[]
pos=[]
for sent in nlp.pipe(df['Sentence']):
    if sent.has_annotation("DEP"):
        token.append([word.text for word in sent])
        pos.append([word.pos_ for word in sent])


# In[6]:


token


# In[7]:


pos


# In[9]:


df['token']=token
df['pos']=pos


# In[10]:


df.head


# In[11]:


df['noun']=df.apply(lambda x: x['pos'].count('NOUN'), axis=1)
df['verb']=df.apply(lambda x: x['pos'].count('VERB'), axis=1)
df['adj']=df.apply(lambda x: x['pos'].count('ADJ'), axis=1)
df['punct']=df.apply(lambda x: x['pos'].count('PUNCT'), axis=1)
df


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv(r"C:\Users\prasa\Downloads\tripadvisor_hotel_reviews.csv")
df.head(22)


# In[3]:


def ratings(rating):
    if rating>3 and rating<=5:
        return "Positive"
    if rating>0 and rating<=3:
        return "Negative"


# In[5]:


import pandas as pd

data = {'Rating': [4, 5, 3, 4, 2]}
df = pd.DataFrame(data)


# In[12]:


import matplotlib.pyplot as plt
rating_counts = df['Rating'].value_counts()
plt.pie(rating_counts, labels=rating_counts.index.tolist(), autopct='%1.1f%%')
plt.show()


# In[13]:


length=len(df['Review'][0])
print(f'length of a sample review: {length}')


# In[14]:


df['length']=df['Review'].str.len()
df.head()


# In[15]:


word_count=df['Review'][0].split()
print(f'word count in a sample review: {len(word_count)}')


# In[12]:


def word_count(review):
    review_list=review.split()
    return len(review_list)


# In[20]:


df['Word_count'] = df['Review'].apply(lambda rev: len(rev.split()))  # Count words
df['mean_word_length'] = df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))  # Mean word length


# In[29]:


df['mean_word_length']=df['Review'].map(lambda rev:np.mean([len(word)for word in rev.split()]))
df.head()


# In[25]:


from nltk.tokenize import sent_tokenize
import numpy as np
mean_sentence_length = np.mean([len(sent) for sent in sent_tokenize(df['Review'][0])])
df['mean_sent_length'] = df['Review'].map(lambda rev: np.mean([len(sent) for sent in sent_tokenize(rev)]))


# In[26]:


features=df.columns.tolist()[2:]
df=df.drop(features,axis=1)
df.head()


# In[31]:


import nltk
nltk.download('stopwords')


# In[32]:


import re
from nltk.corpus import stopwords
def clean(review):
    review=review.lower()
    review=re.sub('[^a-z A-Z 0-9-]+', '', review)
    revi=" ".join([word for word in review.split() if word not in stopwords.words('english')])
    return review


# In[33]:


df['Review']=df['Review'].apply(clean)
df.head()


# In[ ]:


from tqdm import trange
corpus=[]
for i in trange(df.shape[0], ncols=150, nrows=10, colour='green', smoothing=0.8):
    corpus += df['Review_lists'] [i]
len(corpus)


# In[4]:


doc_trump = '"Mr. Trump became President after winning the political election.Though he lost the support of some republican friends, Tump is friends with President Putin"'

doc_election = "President Trump says putin had no political interference is the election outcome.He says it was a withchhunt by political parties.He claimed President Putin is a friend who had nothing to do with the election"

doc_putin = "Post elections, Vladimir Putin became President of Russia.President Putin had served as the Prime Ministee earlier in his political career"

documents = [doc_trump, doc_election, doc_putin]


# In[5]:


documents


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
documents = ["Trump talks about election", "Putin discusses policies", "The election was controversial"]
count_vect = CountVectorizer(stop_words='english')
sparse_matrix = count_vect.fit_transform(documents)
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, columns=count_vect.get_feature_names_out(), index=['doc_trump', 'doc_election', 'doc_putin'])
print(df)


# In[7]:


from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(df,df))


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
# consider two vectors A and B in 2-D
A = np.array([7,3])
B = np.array([3,7])
ax = plt.axes()
ax.arrow(0.0, 0.0, A[0], A[1], head_width=0.4, head_length=0.05, fc='r', ec='r')
plt.annotate(f"A({A[0]},{A[1]})", xy=(A[0], A[1]),xytext=(A[0]+0.1, A[1]+0.1))
ax.arrow(0.0, 0.0, B[0], B[1], head_width=0.4, head_length=0.05, fc='b', ec='b')
plt.annotate(f"B({B[0]},{B[1]})", xy=(B[0], B[1]),xytext=(B[0]+0.1, B[1]+0.1))
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()
plt.close()

cos_sim = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
print (f"Cosine Similarity between A and B:{cos_sim}")
print (f"Cosine Distance between A and B:{1-cos_sim}")


# In[9]:


A = {1,2,3,4,6}
B = {1,2,5,8,9}
C = A.intersection(B)
D = A.union(B)
print('AnB = ', C)
print('AUB = ', D)
print('J(A,B) = ', float(len(C))/float(len(D)))


# In[10]:


def jaccard_similarity(set1, set2):
  intersection = len(set1.intersection(set2))
  union = len(set1.union(set2))
  return intersection / union

set_a = {"Language", "for", "Computer", "NLP", "Science"}
set_b = {"NLP", "for", "Language", "Data", 'ML', "AI"}
similarity = jaccard_similarity(set_a, set_b)
print("Jaccard Similarity:", similarity)


# In[3]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[4]:


responses=[
    "You can return an item within 7 days of purchase.",
    "Our return policy allows you to return items that are unopened and in their original condition.",
    "We offer free shipping on orders over $50.",
    "To track your order,you can visit.",
    "Our customer support team is available from 9am-6pm.Monday to Friday."
]


# In[5]:


user_input="How can I track my order?"


# In[6]:


vectorizer=TfidfVectorizer(stop_words='english')
all_texts=responses+[user_input]


# In[7]:


tfidf_matrix=vectorizer.fit_transform(all_texts)


# In[8]:


user_vector=tfidf_matrix[-1]
response_vectors=tfidf_matrix[:-1]
cosine_similarities=cosine_similarity(user_vector, response_vectors)


# In[9]:


cosine_similarities=cosine_similarity(user_vector, response_vectors)


# In[10]:


most_similar_idx=np.argmax(cosine_similarities)


# In[11]:


print(f"User Query: {user_input}")
print(f"Most relevant response: {responses[most_similar_idx]}")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import random
import string
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer


# In[2]:


pip install nltk


# In[3]:


import nltk


# In[4]:


nltk.download('popular' ,quiet=True)
nltk.download('punkt')
nltk.download('wordnet')


# In[ ]:


lemmer=nltk.stem.WordNetLemmatizer()
def LemZTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict=dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[ ]:


GREETING_INPUTS=("hello", "hi", "greetings", "what's up hey",\"how are you?")
GREETINGS_RESPONSES=["hi", "hey", "hi there", "hello",/"I am glad! you are talking to me", \"I am fine! How about you?"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETINGS_RESPONSES)


# In[6]:


def response(user_response):
    robo_response = ""
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand."
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


# In[9]:


flag=True
print("SABot: My name is SABot. How can I assist you?. \
If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response == 'thank you' ):
            flag=False
            print("SABot: You are welcome...")
        else:
            if(greeting(user_response)!=None):
                print("SABot: "+greeting(user_response))
            else:
                print("SABot: ",end=" ")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("SABot: Bye! take care...")


# In[ ]:





# In[ ]:





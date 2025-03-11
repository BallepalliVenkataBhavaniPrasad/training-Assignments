#!/usr/bin/env python
# coding: utf-8

# In[1]:


sent="Ram is studying at Malla Reddy University in Hyderabad, India"


# In[2]:


import nltk
nltk.download('words')


# In[3]:


get_ipython().system('pip install svgling')


# In[4]:


import nltk
from nltk import ne_chunk
from nltk import word_tokenize
ne_chunk(nltk.pos_tag(word_tokenize(sent)), binary=False)


# In[6]:


get_ipython().system('pip install spacy')


# In[10]:


import spacy
nlp=spacy.load('en_core_web_sm')
doc=nlp(u'Apple is ready to launch new phone worth $10000 in New York time square ')
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


# In[11]:


import spacy
nlp=spacy.load('en_core_web_sm')


# In[12]:


text="""Elon Musk, the CEO of SpaceX and Tesla, announced that SpaceX's Starship will be launching its first crewed mission to Mars in 2027.
The mission, which will involve astronauts from NASA, will be the first of its kind, and it will take place at the Kennedy Space Center in Florida.
Musk emphasized that the project would push the boundaries of space exploration."""


# In[13]:


doc=nlp(text)
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")


# In[14]:


from spacy import displacy
displacy.render(doc, style="ent")


# In[15]:


import pandas as pd
entities=[(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]
df=pd.DataFrame(entities, columns=['text', 'type', 'lemma'])
print(df)


# In[ ]:





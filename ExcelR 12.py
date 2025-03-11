#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().system('pip install pytextrank')


# In[12]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[13]:


import spacy
import pytextrank


# In[14]:


document = """Not only did it only confirm that the film would be unfunny
and generic, but it also managed to give away the ENTIRE movie;
and I'm not exaggerating - every moment, every
plot point, every joke is told in the trailer."""


# In[15]:


en_nlp=spacy.load("en_core_web_sm")
en_nlp.add_pipe("textrank")
doc=en_nlp(document)


# In[16]:


tr=doc._.textrank
print(tr.elapsed_time)


# In[19]:


for combination in doc._.phrases:
    print(combination.text, combination.rank, combination.count)


# In[20]:


from bs4 import BeautifulSoup
from urllib.request import urlopen


# In[21]:


def get_only_text(url):
    page=urlopen(url)
    soup=BeautifulSoup(page)
    text='\t'.join(map(lambda p: p.text, soup.find_all('p')))
    print(text)
    return soup.title.text, text


# In[22]:


url="https://en.wikipedia.org/wiki/Natural_language_processing"
text=get_only_text(url)


# In[23]:


len(''.join(text))


# In[24]:


text[:1000]


# In[8]:


get_ipython().system('pip install sumy')


# In[4]:


get_ipython().system('pip install lxml.html.clean')


# In[5]:


from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.luhn import LuhnSummarizer


# In[7]:


LANGUAGE="english"
SENTENCES_COUNT=3
url="https://en.wikipedia.org/wiki/Natural_language_processing"
parser=HtmlParser.from_url(url, Tokenizer(LANGUAGE))
summarizer=LsaSummarizer(Stemmer(LANGUAGE))
summarizer.stop_words=get_stop_words(LANGUAGE)
for sentence in summarizer(parser.document, 5):
    print(sentence)


# In[25]:


text="""A vaccine for the coronavirus will likely be ready by early 2021 but rolling it out safely across India's 1.3 billion people will be the country's biggest challenge in fighting its surging epidemic, a leading vaccine scientist told Bloomberg.
India, which is host to some of the front-runner vaccine clinical trials, currently has no local infrastructure in place to go beyond immunizing babies and pregnant women, said Gagandeep Kang, professor of microbiology at the Vellore-based Christian Medical College and
a member of the WHO's Global Advisory Committee on Vaccine Safety.
The timing of the vaccine is a contentious subject around the world. In the U.S., President Donald Trump has contradicted a top administration health expert by saying a vaccine would be available by October. In India, Prime Minister Narendra Modi's government had
promised an indigenous vaccine as early as mid-August, a claim the government and its apex medical research body has since walked back.
"""


# In[26]:


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer


# In[27]:


parser = PlaintextParser.from_string(text, Tokenizer("english"))


# In[28]:


from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.utils import get_stop_words
summarizer_lex = LexRankSummarizer()
summarizer_lex.stop_words = get_stop_words("english")
summary= summarizer_lex(parser.document, 5)
lex_summary=""
for sentence in summary:
    lex_summary+=str(sentence)
print(lex_summary)


# In[ ]:





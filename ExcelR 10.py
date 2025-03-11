#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install PyPDF2')


# In[3]:


import PyPDF2
from PyPDF2 import PdfFileReader


# In[5]:


PyPDF2.__version__


# In[6]:


import PyPDF2, urllib, nltk
from io import BytesIO
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[8]:


wFile=urllib.request.urlopen('https://www.udri.org/pdf/02%20working%20paper%201.pdf')
pdfreader=PyPDF2.PdfReader(BytesIO(wFile.read()))


# In[9]:


pageObj=pdfreader.pages[2]
page2=pageObj.extract_text()
punctuations=['(',')',';','[',']',',','...','.']
tokens=word_tokenize(page2)
stop_words=stopwords.words('english')
keywords=[word for word in tokens if not word in stop_words and not word in punctuations]


# In[10]:


keywords


# In[13]:


name_list=list()
check=['Mr.','Mrs.','Ms.']
for idx,token in enumerate(tokens):
    if token.startswith(tuple(check)) and idx < (len(tokens)-1):
        name=token+tokens[idx+1]+' '+tokens[idx+2]
        name_list.append(name)
print(name_list)


# In[14]:


wFile.close()


# In[16]:


pdf=open(r"C:\Users\prasa\Downloads\02 working paper 1.pdf","rb")
pdf_reader=PyPDF2.PdfReader(pdf)
print("Number of pages:",len(pdf_reader.pages))
page=pdf_reader.pages[1]
print(page.extract_text())
pdf.close


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import math


# In[11]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


# In[3]:


text = "Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves the development of algorithms and models to understand, analyze, and generate human language."


# In[4]:


#sentence tokenization
from nltk.tokenize import sent_tokenize
sentencetokenize=sent_tokenize(text)
print(sentencetokenize)


# In[5]:


#word tokenization
from nltk.tokenize import word_tokenize
wordtoken = word_tokenize(text.lower())
print(wordtoken)


# In[7]:


#stop word removal
from nltk.corpus import stopwords
stop_words=set(stopwords.words('English'))
symbols=['(',')',',','.']
filteredtext=[]
for i in wordtoken:
    if i not in stop_words and i not in symbols:
        filteredtext.append(i)
print(filteredtext)


# In[8]:


#stemming
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed=[]
for i in filteredtext:
    stemmed.append(ps.stem(i))
print(stemmed)


# In[12]:


#Lemmatiztion

from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
lemmatized=[]
for i in filteredtext:
    lemmatized.append(wnl.lemmatize(i))
print(lemmatized)


# In[13]:


#POS Tagging
pos_tags=[]
pos_tags.extend(nltk.pos_tag(wordtoken))
for word, pos_tag in pos_tags:
    print(f"{word}: {pos_tag}")


# In[15]:


#TF-IDF
Tf={word: filteredtext.count(word)/len(filteredtext) for word in filteredtext}
Tf


# In[17]:


noofdocuments=1
IDF={word:math.log(noofdocuments/filteredtext.count(word)+1) for word in filteredtext}
IDF


# In[18]:


tfidf = {word:Tf[word]*IDF[word] for word in filteredtext}
tfidf


# In[ ]:





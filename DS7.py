#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('average_perceptron_tagger')
nltk.download('wordnet')


# In[3]:


sample_document="Text analytics is the process of converting unstructured text data into meaningful and actionable information."

tokens= word_tokenize(sample_document)
print("Tokenization",tokens)


# In[6]:


pos_tags = pos_tag(tokens)
print('POS Tagging:',pos_tags)


# In[7]:


stop_words=set(stopwords.words('english'))
filtered_tokens= [word for word in tokens if word.lower() not in stop_words]
print("Stop Words: ",filtered_tokens)


# In[9]:


stemmer=PorterStemmer()
stemmed_tokens= [stemmer.stem(word) for word in filtered_tokens]
print("Stemming:",stemmed_tokens)


# In[10]:


lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("Lemmatization",lemmatized_tokens)


# In[11]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_representation = tfidf_vectorizer.fit_transform([sample_document])
print('TF-IDF Representation')
print(tfidf_representation)


# In[ ]:





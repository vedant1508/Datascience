#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns



# In[7]:


df = pd.read_csv("IRIS.csv")


# In[8]:


df


# In[9]:


df.describe()


# In[10]:


features = df.iloc[:,1:5]
datatypes = df.iloc[:,1:5].dtypes
datatypes


# In[12]:


df.head()


# In[13]:


datatypes.head()


# In[15]:


sns.histplot(df['sepal_length'],kde=True)


# In[17]:


sns.histplot(df['sepal_width'],kde=True)


# In[18]:


sns.histplot(df['petal_length'],kde=True)


# In[19]:


sns.histplot(df['petal_width'],kde=True)


# In[22]:


sns.boxplot(df['sepal_length'])


# In[23]:


sns.boxplot(df['sepal_width'])


# In[24]:


sns.boxplot(df['petal_length'])


# In[26]:


sns.boxplot(df['petal_width'])


# In[ ]:





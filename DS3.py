#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[6]:


data = ({
    "Age":np.random.choice(['Young','Adult','Mid-Age','Old'],50),
    "Income":np.random.randint(20000,50000,50)
})


# In[8]:


df = pd.DataFrame(data)


# In[9]:


df.head()


# In[10]:


df.describe()


# In[11]:


summary_stats = df.groupby('Age')['Income'].agg(['mean','median','min','max','std'])


# In[12]:


summary_stats


# In[14]:


df2=pd.read_csv('Iris.csv')


# In[15]:


df2.head()


# In[16]:


df2.describe()


# In[19]:


group=df2.groupby('Species')


# In[21]:


iris_versicolor=group.get_group('Iris-versicolor')
iris_versicolor
iris_versicolor.describe()


# In[ ]:





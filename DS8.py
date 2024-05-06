#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = sns.load_dataset('titanic')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.drop('deck',axis=1,inplace=True)


# In[9]:


df.info()


# In[11]:


nullvals = np.where(df['age'].isnull())[0]
for i in nullvals:
    df.drop(i,inplace=True)


# In[12]:


df.info()


# In[16]:


counts = df['who'].value_counts()
sns.set_style('darkgrid')
sns.set_context("notebook",font_scale=1.2)
plt.pie(counts,labels=counts.index,autopct='%1.1f%%',startangle=90)
plt.axis('equal')
plt.show()


# In[17]:


sns.histplot(df['age'])


# In[18]:


sns.relplot(data=df,x='age',y='fare',hue='fare')


# In[19]:


sns.histplot(df['fare'])
plt.title('Fare vs. Count')


# In[20]:


df.groupby('embark_town').describe()['survived']


# In[ ]:





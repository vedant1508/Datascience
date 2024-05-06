#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[23]:


df = pd.read_csv('StudentsPerformance1.csv')


# In[24]:


df


# In[25]:


df.shape


# In[26]:


#Handel Missing values and inconsistencies

df.isnull().sum()


# In[29]:


df['math score'].fillna(df['math score'].mean(),inplace=True)  
df['reading score'].fillna(df['reading score'].mean(),inplace=True)
df['writing score'].fillna(df['writing score'].mean(),inplace=True)


# In[30]:


df.isna().sum()


# In[32]:


sns.boxplot(df[['math score','reading score','writing score']])


# In[38]:


#Address outliers using Winsoriztion

from scipy.stats.mstats import winsorize
df['math score']=winsorize(df['math score'],limits=[0.05,0.05])
df['reading score']=winsorize(df['reading score'],limits=[0.05,0.05])
df['writing score']=winsorize(df['writing score'],limits=[0.05,0.05])


# In[39]:


sns.boxplot(df[['math score','reading score','writing score']])


# In[40]:


from sklearn.preprocessing import MinMaxScaler


# In[41]:


scaler= MinMaxScaler()


# In[42]:


df[['math score']] = scaler.fit_transform(df[['math score']])


# In[43]:


df


# In[ ]:





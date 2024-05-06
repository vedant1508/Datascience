#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv('StudentsPerformance.csv')


# In[7]:


df.head()  #prints first 5 rows
#df.tail() prints last 5 rows


# In[8]:


df.dtypes  #print the datatype of data in each column here object type stands for string data


# In[10]:


df.describe()
# used to generate descriptive statistics of a DataFrame.
# it provides a summary of the central tendency, dispersion, and shape of the numerical columns in the DataFrame.
#count- number of non null values, mean- average,std - standard deviation, quartile ranges and min and max of column


# In[14]:


df.shape #prints tuple of rows and columns present in df 
            #df.size prints rows * columns or total number of values in the df


# In[15]:


df.ndim #prints the dimensions of the data


# In[17]:


df.columns #prints the list of columns in the dataframe


# In[18]:


df.mean()


# In[19]:


df.isna().sum()


# In[21]:


df['math score']=df['math score'].fillna(df['math score'].mean())
df['reading score']=df['reading score'].fillna(df['reading score'].mean())
df['writing score']=df['writing score'].fillna(df['writing score'].mean())


# In[22]:


df.isnull().sum()


# In[23]:


df[df['reading score']>75]


# In[25]:


df.loc[60:80,['math score']] #in .loc 60 and 80 both will be considered


# In[28]:


df.iloc[60:80,[5]] #printing individual columns


# In[30]:


df['math score'].skew()
#prints the skewness value
#skewness- It provides information about the shape of the distribution. 
#the skewness refers to the asymmetry of the distribution of the data points along the horizontal axis.
#positive,negative and zero skewness
# Skewness = (3 * (mean - median)) / standard deviation  -> pearson coefficient of skewness


# In[32]:


df['math score'].kurt()
# Kurtosis-It provides information about the presence of extreme values or outliers in the data and the concentration
# of data around the mean.
#leptokurtic, mesokurtic, platykurtic


# In[36]:


df['math score'].var() #variance
#variance-Variance is a statistical measure that quantifies the spread or dispersion of a set of data points. 
# It measures how far individual data points in a dataset are spread out from the mean or average value


# In[ ]:


#Converting categorical data to quantitative data


# In[37]:


from sklearn.preprocessing import LabelEncoder


# In[39]:


encoder = LabelEncoder()
df['gender']= encoder.fit_transform(df['gender'])
df['lunch']=encoder.fit_transform(df['lunch'])


# In[40]:


df


# In[ ]:





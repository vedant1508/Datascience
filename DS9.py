#!/usr/bin/env python
# coding: utf-8

# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df = sns.load_dataset('titanic')


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


sns.boxplot(data=df,x='sex',y='age',hue='survived')
plt.title("Distribution of Age by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")


# In[10]:


Observations:

1.The median age for male passengers who survived was lower than that of female passengers who survived.The age distribution for male passengers who did not survive was wider than that of male passengers who survived. 
2. The age distribution for female passengers who did not survive was wider than that of female passengers who survived. 
3. There were some outliers in the age distribution for both male and female passengers who did not survive. 
4. Overall, there appears to be a slight difference in the age distribution between male and female passengers who survived, with female passengers generally being older. However, the age distribution for both genders is similar for passengers who did not survive.


# In[ ]:





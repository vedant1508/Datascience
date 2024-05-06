#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# In[41]:


df = pd.read_csv('boston.csv')


# In[42]:


df.head()


# In[43]:


df.columns


# In[44]:


df.isnull().sum()


# In[45]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[46]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25, random_state=42)


# In[47]:


model = LinearRegression()
model.fit(x_train,y_train)


# In[49]:


y_pred = model.predict(x_test)


# In[50]:


print('MSE:',metrics.mean_squared_error(y_test, y_pred))


# In[38]:


model.score(x_train,y_train)


# In[39]:


model.score(x_test,y_test)


# In[27]:


model.predict([[0.03237,0.0,2.18,0,0.458,6.998,45.8,6.0622,3,222.0,18.7,394.63,2.94]])


# In[ ]:





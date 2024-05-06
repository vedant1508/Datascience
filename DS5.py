#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error


# In[47]:


df = pd.read_csv("Social_Network_Ads.csv")


# In[48]:


df.head()


# In[49]:


df.shape


# In[50]:


df.columns


# In[51]:


df.info()


# In[52]:


df.describe()


# In[53]:


df.isnull().sum()


# In[54]:


x = df.iloc[:,[2,3]]
y = df.iloc[:,-1]


# In[55]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[56]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[57]:


le = LabelEncoder()
df['Gender']= le.fit_transform(df['Gender'])


# In[58]:


df.head()


# In[59]:


regression = LogisticRegression()


# In[60]:


regression.fit(x_train,y_train)


# In[61]:


y_pred = regression.predict(x_test)


# In[62]:


regression.score(x_test, y_test)


# In[63]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[64]:


print('MSE: ',metrics.mean_squared_error(y_test, y_pred))


# In[65]:


cm = metrics.confusion_matrix(y_test,y_pred)


# In[67]:


#Calucaltion TP,FN,TN,FP

TP = cm[0][0]
TN = cm[0][1]
FP = cm[1][0]
FN = cm[1][1]

print('TP',TP)
print('FN',FN)
print('TN',TN)
print('FP',FP)


# In[68]:


#Calculating Accuracy, Error rate, Precision, Recall

accuracy = TP+TN/(TP+FN+TN+FP)
error_rate = FP+FN/(TP+FN+TN+FP)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print("Accuracy",accuracy)
print("Error Rate",error_rate)
print("Precision",precision)
print("Recall",recall)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# In[4]:


df = pd.read_csv('iris.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# In[14]:


x = df.iloc[:,1:5]
y = df.iloc[:,-1]
y


# In[15]:


sc = StandardScaler()
X =sc.fit_transform(x)


# In[16]:


X


# In[17]:


x_train, x_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)


# In[19]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[20]:


nb = GaussianNB()


# In[21]:


nb.fit(x_train,y_train)


# In[22]:


y_pred = nb.predict(x_test)


# In[23]:


nb.score(x_test,y_test)*100


# In[26]:


cm = confusion_matrix(y_test,y_pred)

TP= cm[0][0]
TN= cm[0][1]
FP= cm[1][0]
FN= cm[1][1]

print('TP',TP)
print('FN',FN)
print('TN',TN)
print('FP',FP)


# In[27]:


accuracy = TP+TN/(TP+TN+FP+FN)
error_rate = FP+FN/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print("Accuracy",accuracy)
print("Error Rate",error_rate)
print("Precision",precision)
print("Recall",recall)


# In[ ]:





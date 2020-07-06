#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv('../data-files/appended_songs.csv')


# In[4]:


dataset.head()


# In[5]:


X = dataset.iloc[:,4:].values
y = dataset.iloc[:,2].values


# In[6]:


X[0:5,:]


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0)


# In[13]:


from sklearn.cross_validation import train_test_split


# In[10]:


import sklearn


# In[11]:


import sklearn


# In[ ]:





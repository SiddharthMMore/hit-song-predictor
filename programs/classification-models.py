#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Reading dataset
dataset = pd.read_csv('C:\\Users\\DELL\\Desktop\\ASSIGNMENTS\\minor-project\\data-files\\appended_songs.csv')
dataset = dataset.drop(dataset.columns[0],axis = 1)


# In[5]:


dataset.head()


# In[6]:


#Spilling dataset into features and target variable
X = dataset.iloc[:, 4:15].values
y = dataset.iloc[:, 2].values


# In[7]:


X


# In[8]:


y


# In[9]:


#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[10]:


#Performing feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[12]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, y_train)


# In[14]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifierLR,X = X_train ,y = y_train, cv = 10)


# In[15]:


accuracies.mean()


# In[16]:


accuracies.std()


# In[17]:


y_pred_LR = classifierLR.predict(X_test)


# In[18]:


from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(y_test, y_pred_LR)


# In[20]:


import seaborn as sns
sns.heatmap(cm_LR/np.sum(cm_LR), annot=True, fmt='.2%')


# In[21]:


#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)


# In[22]:


y_pred_NB = classifierNB.predict(X_test)


# In[23]:


cm_NB = confusion_matrix(y_test, y_pred_NB)


# In[29]:


accuraciesNB = cross_val_score(estimator = classifierNB,X = X_train ,y = y_train, cv = 10)


# In[30]:


accuraciesNB.mean()


# In[31]:


accuraciesNB.std()


# In[32]:


sns.heatmap(cm_NB/np.sum(cm_NB), annot=True, fmt='.2%')


# In[33]:


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)


# In[34]:


y_pred_KNN = classifierKNN.predict(X_test)


# In[35]:


accuraciesKNN = cross_val_score(estimator = classifierKNN,X = X_train ,y = y_train, cv = 10)


# In[36]:


accuraciesKNN.mean()


# In[37]:


accuraciesKNN.std()


# In[38]:


cm_KNN = confusion_matrix(y_test, y_pred_KNN)


# In[39]:


sns.heatmap(cm_KNN/np.sum(cm_KNN), annot=True, fmt='.2%')


# In[40]:


#Decesion Tree Classification
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDT.fit(X_train, y_train)


# In[41]:


y_pred_DT = classifierDT.predict(X_test)


# In[42]:


accuraciesDT = cross_val_score(estimator = classifierDT,X = X_train ,y = y_train, cv = 10)


# In[43]:


accuraciesDT.mean()


# In[44]:


accuraciesDT .std()


# In[46]:


cm_DT = confusion_matrix(y_test, y_pred_DT)


# In[47]:


sns.heatmap(cm_KNN/np.sum(cm_KNN), annot=True, fmt='.2%')


# In[48]:


from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierRF.fit(X_train, y_train)


# In[49]:


y_pred_RF = classifierRF.predict(X_test)


# In[50]:


accuraciesRF = cross_val_score(estimator = classifierRF,X = X_train ,y = y_train, cv = 10)


# In[51]:


accuraciesRF.mean()


# In[52]:


accuraciesRF.std()


# In[53]:


cm_RF = confusion_matrix(y_test, y_pred_RF)


# In[54]:


sns.heatmap(cm_RF/np.sum(cm_RF), annot=True, fmt='.2%')


# In[55]:


from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'linear', random_state = 0)
classifierSVM.fit(X_train, y_train)


# In[56]:


y_pred_SVM = classifierSVM.predict(X_test)


# In[57]:


accuraciesSVM = cross_val_score(estimator = classifierSVM,X = X_train ,y = y_train, cv = 10)


# In[58]:


accuraciesSVM.mean()


# In[59]:


accuraciesSVM.std()


# In[60]:


cm_SVM = confusion_matrix(y_test, y_pred_SVM)


# In[61]:


sns.heatmap(cm_SVM/np.sum(cm_SVM), annot=True, fmt='.2%')


# In[62]:


#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
classifierANN = Sequential()

classifierANN.add(Dense(6,input_dim = 11,activation = 'relu' ))
classifierANN.add(Dense(6,activation = 'relu'))
classifierANN.add(Dense(1,activation = 'sigmoid'))


# In[63]:


classifierANN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[64]:


history = classifierANN.fit(X_train, y_train, epochs=100, batch_size=64)


# In[66]:


y_pred_ANN = classifierANN.predict(X_test)
y_pred_ANN = (y_pred_ANN > 0.5)


# In[67]:


cm_ANN = confusion_matrix(y_test, y_pred_ANN)


# In[68]:


sns.heatmap(cm_ANN/np.sum(cm_ANN), annot=True, fmt='.2%')


# In[ ]:





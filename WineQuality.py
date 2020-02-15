#!/usr/bin/env python
# coding: utf-8

# In[176]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# In[4]:


data = pd.read_csv('winequalityN.csv')


# In[24]:


data.head()
#data.describe()
data.drop(['type'],axis=1,inplace=True)


# In[123]:


data.head()
a = data['fixed acidity'].median()
data['fixed acidity'].fillna(a,inplace=True)
b = data['volatile acidity'].median()
data['volatile acidity'].fillna(b,inplace=True)
c = data['citric acid'].median()
data['citric acid'].fillna(c,inplace=True)
d = data['residual sugar'].median()
data['residual sugar'].fillna(d,inplace=True)
e = data['chlorides'].median()
data['chlorides'].fillna(e,inplace=True)
f = data['pH'].median()
data['pH'].fillna(f,inplace=True)
g = data['sulphates'].median()
data['sulphates'].fillna(g,inplace=True)


# In[124]:


list1 = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
x =data[list1]
y = data['quality']


# In[158]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.25)


# In[180]:


model = KNeighborsClassifier()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
df = pd.DataFrame({'actual':ytest,'ypred':ypred})
df
model.score(xtest,ytest)


# In[150]:


mod = GaussianNB()
mod.fit(xtrain,ytrain)
mod.predict(xtest)
mod.score(xtest,ytest)


# In[151]:


mod1 = SVC()
mod1.fit(xtrain,ytrain)
mod1.predict(xtest)
mod1.score(xtest,ytest)


# In[178]:


mod2 = DecisionTreeClassifier(max_depth=2)
mod2.fit(xtrain,ytrain)
mod2.predict(xtest)
mod2.score(xtest,ytest)


# In[169]:


mod3 = RandomForestClassifier(n_estimators=700)
mod3.fit(xtrain,ytrain)
mod3.predict(xtest)
mod3.score(xtest,ytest)


# In[179]:


plot_tree(mod2.fit(xtrain,ytrain))


# In[ ]:





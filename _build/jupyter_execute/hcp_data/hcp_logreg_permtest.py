#!/usr/bin/env python
# coding: utf-8

# # Permutation Test

# In[1]:


import sys
import numpy as np
import scipy as scp
import sklearn
import pickle
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt


# In[8]:


get_ipython().run_line_magic('store', '-r X_test')
get_ipython().run_line_magic('store', '-r y_test')
get_ipython().run_line_magic('store', '-r logmodel')
get_ipython().run_line_magic('store', '-r logperformAcc')


# In[9]:


# Shuffle each column of X_test to create permutation
def create_permutation(X):
    new_X = np.empty(X.shape)
    for i in range(X.shape[1]):
        randCol = X[:, i]
        np.random.shuffle(randCol)
        new_X[:, i] = randCol
    return new_X


# In[10]:


# Find p value
def findP(t, arr):
    count = 0
    while count < len(arr) and arr[count] > t:
        count += 1
    p = count / len(arr)
    return p


# In[11]:


# Take 200 resamples
acc = []
upper_acc = []
p_vals = []
startindex = 0
endindex = 0
for t in range(90):
    t_acc = []
    while endindex < y_test.shape[0] and int(y_test[endindex, 1]) == t:
        endindex += 1
    X_c = np.copy(X_test[startindex: endindex,])
    for i in range(200):
        new_X = create_permutation(X_c)
        a = logmodel.score(new_X, y_test[startindex:endindex, 0])
        t_acc.append(a)
    startindex = endindex
    t_acc = np.array(t_acc)

    t_acc = sorted(t_acc, reverse = True)
    p = findP(logperformAcc[t], t_acc)
    p_vals.append(p)
    
    upper_acc.append(np.percentile(t_acc, 95))


# In[12]:


get_ipython().run_line_magic('store', 'upper_acc')


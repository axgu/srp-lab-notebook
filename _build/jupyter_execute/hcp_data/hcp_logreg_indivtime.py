#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression - ROI Features

# In[1]:


import numpy as np
import pickle
import scipy as scp
import sklearn
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt


# In[2]:


with open('HCP_movie_watching.pkl','rb') as f:
    TS = pickle.load(f)

index = np.arange(176)
np.random.shuffle(index)
testIndex = index[:76]


# In[3]:


def splitData(tList):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for row in tList:
        if row[-1] in testIndex:
            X_test.append(row[:-3])
            y_test.append(row[-3:-1])
        else:
            X_train.append(row[:-3])
            y_train.append(row[-3:-1])

    X_train = np.array(X_train).astype(float)
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)

    X_test = np.array(X_test).astype(float)
    X_test = (X_test - np.mean(X_test)) / np.std(X_test)
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test
    


# In[4]:


model = LogisticRegression(max_iter = 1000)
arr = []
for k in range(90):
    for key, val in TS.items():
        if val.shape[-2] > k:   # Account for clips with less than 90 time points
            if key == 'testretest':
                for i in range(val.shape[0]):
                    for j in range(val.shape[-3]):
                        subj = []       # Create new row
                        for l in range(val.shape[-1]):
                            subj.append(val[i][j][k][l])
                        subj.append(key)    # Add movie
                        subj.append(k)
                        subj.append(j)      # Add participant number
                        arr.append(subj)     # Add new row to array
            else:
                for j in range(val.shape[-3]):
                    subj = []
                    for l in range(val.shape[-1]):
                        subj.append(val[j][k][l])
                    subj.append(key)
                    subj.append(k)
                    subj.append(j)
                    arr.append(subj)


# In[12]:


X_train, X_test, y_train, y_test = splitData(arr)
model.fit(X_train, y_train[:, 0])


# In[13]:


performAcc = []
startindex = 0
endindex = 0
for t in range(90):
    while endindex < y_test.shape[0] and int(y_test[endindex, 1]) == t:
        endindex += 1
    acc = model.score(X_test[startindex:endindex,], y_test[startindex:endindex, 0])
    performAcc.append(acc)
    startindex = endindex


# In[14]:


xAx = [i for i in range(0,90)]
plt.plot(xAx, performAcc, label="log-reg")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xlim(0,90)
plt.title("Time-varying Classification Accuracy")
plt.legend()
plt.show()


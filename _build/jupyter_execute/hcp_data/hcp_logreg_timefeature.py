#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression - (Time point, ROI) Features

# In[1]:


import numpy as np
import pickle
import scipy as scp
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[2]:


# Turn dictionary into 2D Array
def createData(movieDict):
    # movieList = list(movieDict.keys())
    # vals = list(movieDict.values())
    
    # Reduce to 2 dimensions
    X = np.empty((176*18, 65*300+2), dtype="object")

    rCount = 0
    for key, row in movieDict.items():
        # Testretest
        if len(row.shape) == 4:
            for i in range(row.shape[0]):
                for j in range(row.shape[-3]):
                    X[rCount][-2] = 'testretest'
                    X[rCount][-1] = j
                    for k in range(65):
                        for l in range(row.shape[-1]):
                            X[rCount][k*row.shape[-1] + l] = row[i][j][k][l]
                    rCount += 1
                            
        # Otherwise
        else:
            for j in range(row.shape[-3]):
                X[rCount][-2] = key
                X[rCount][-1] = j
                for k in range(65):
                    for l in range(row.shape[-1]):
                        X[rCount][k*row.shape[-1] + l] = row[j][k][l]
                rCount += 1
                         
    # Randomly split participants
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    index = np.arange(176)
    np.random.shuffle(index)
    testIndex = index[:76]

    np.random.shuffle(X)
    for row in X:
        if row[-1] in testIndex:
            X_test.append(row[:-2])
            y_test.append(row[-2])
        else:
            X_train.append(row[:-2])
            y_train.append(row[-2])

    X_train = np.array(X_train).astype(float)
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test = np.array(X_test).astype(float)
    X_test = (X_test - np.mean(X_test)) / np.std(X_test)
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test
    


# In[3]:


with open('HCP_movie_watching.pkl','rb') as f:
    TS = pickle.load(f)


# In[4]:


# accList = []
X_train, X_test, y_train, y_test = createData(TS)

model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
# accList.append(acc)
print(acc)


# In[ ]:


# Logistic Regression
def cost(X, Y, W):
    h = 1 / (1 + np.exp(-np.dot(X, W))) # hypothesis representation
    cost = np.dot(Y, -np.log(h)) + np.dot((1-Y), np.log(1-h)) # cost function
    J = -1 / (len(X)) * np.sum(cost) # mean cost
    return J


def gradient(X, Y, W):
    h = 1 / (1 + np.exp(-np.dot(X, W)))
    diff = h - Y
    grad = 1 / (len(X)) * np.dot(diff, X)
    return grad

    
def descent(X_train, Y_train, lr = 0.01):
    weights = [0]*(len(X_train[0]))
    loss = []
    loss.append(cost(X_train, Y_train, weights))
    count = 0
    while count < 1000:
        grad = gradient(X_train, Y_train, weights)
        weights = weights - lr*grad
        loss.append(cost(X_train, Y_train, weights))
        count += 1

    return weights

def createYMask(movie, Y):
    yMasked = np.zeros(Y.shape)
    mask = Y == movie
    yMasked[mask] = 1
    return yMasked
    
def sigmoid(X, W):
    return 1 / (1 + np.exp(-np.dot(X, W)))

"""
movieList = list(TS.keys())
modelWeights = []
for movie in movieList:
    yMasked = createYMask(movie, y_train)
    W = descent(X_train, yMasked)
    modelWeights.append(W)
predY = []
for x in X_test:
    probList = [sigmoid(x, W) for W in modelWeights]
    predY.append(movieList[probList.index(max(probList))])

pMask = y_test == predY # create mask for values where predicted is correct
acc = sum(pMask) / len(pMask)
print(acc)
"""


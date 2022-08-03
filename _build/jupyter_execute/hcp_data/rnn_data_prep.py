#!/usr/bin/env python
# coding: utf-8

# # LSTM Model

# In[1]:


import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, LSTM, Layer
from keras import Input


# In[2]:


with open('HCP_movie_watching.pkl','rb') as f:
    TS = pickle.load(f)

get_ipython().run_line_magic('store', '-r testIndex')


# In[3]:


def splitDict(dict, indexList = testIndex):
    trainDict = {}
    testDict = {}
    for key, val in dict.items():
        testvalArr = []
        trainvalArr = []
        if key == "testretest":
            testvalArr = testvalArr
            for i in range(val.shape[0]):
                for p in range(val.shape[1]):
                    if p in testIndex:
                        testvalArr.append(val[i][p])
                    else:
                        trainvalArr.append(val[i][p])
        else:
            for p in range(val.shape[0]):
                if p in testIndex:
                        testvalArr.append(val[p])
                else:
                    trainvalArr.append(val[p])
        trainDict[key] = np.array(trainvalArr)
        testDict[key] = np.array(testvalArr)
    return trainDict, testDict


# In[4]:


def vectorize_labels(labels, keys, classNums = 13):
    results = np.zeros((int(labels.shape[0]), int(labels.shape[1]), classNums))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] != -100.:
                indexNum = keys.index(labels[i][j])
                results[i][j][indexNum] = 1.
    return results


# In[5]:


# Reshape data into 3D array

def shaping(dict):
    X_arr = []
    y_arr = []
    keylist = list(dict.keys())
    for key, val in dict.items():
        for i in range(val.shape[0]):
            normalized_seq = (val[i] - np.mean(val[i])) / np.std(val[i])
            X_arr.append(normalized_seq)
            clip = [key for j in range(min(val.shape[1], 90))]
            y_arr.append(clip)
        
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_arr, maxlen = 90, dtype='float64', padding='post', truncating='post')
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(y_arr, maxlen = 90, dtype = 'object', padding='post', truncating='post', value = -100.)
    y_padded = vectorize_labels(y_padded, keylist, classNums = len(keylist))

    return tf.convert_to_tensor(X_padded, dtype='float64'), tf.convert_to_tensor(y_padded)


# In[6]:


train, test = splitDict(TS)
X_train, y_train = shaping(train)
X_test, y_test = shaping(test)


# In[17]:


def createMask(X):
    sample = X.shape[0]
    time_seq = X.shape[1]

    mask = np.empty((sample, time_seq), dtype=np.bool_)
    for i in range(sample):
        for j in range(time_seq):
            if np.count_nonzero(X[i][j]) == 0:
                mask[i][j] = False
            else:
                mask[i][j] = True
    return tf.convert_to_tensor(mask)


# In[18]:


mask = createMask(X_train)


# In[21]:


inputs = Input(shape=(X_train.shape[1],X_train.shape[2]))
x = LSTM(units=32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, dropout=0.05)(inputs)
outputs = TimeDistributed(Dense(15, activation='softmax'))(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[22]:


history = model.fit(X_train, y_train, batch_size = 32, epochs = 60, validation_split=0.2)


# In[23]:


def findIndex(val, arr):
    index = -1
    for x in range(arr.size):
        if val == arr[x]:
            index = x
            break
    return index


# In[24]:


def accuracy(target, prob):
    accArr = []
    for i in range(prob.shape[1]):
        correctCount = 0
        totalCount = 0
        for j in range(prob.shape[0]):
            if np.count_nonzero(target[j][i]) != 0:
                if findIndex(1., target[j][i]) == findIndex(np.amax(prob[j][i]), prob[j][i]):
                    correctCount += 1
                totalCount += 1
        a = correctCount / totalCount
        accArr.append(a)
    return accArr


# In[25]:


pred_test = model.predict(X_test)
acc = accuracy(y_test.numpy(), pred_test)


# In[ ]:





# In[26]:


get_ipython().run_line_magic('store', 'acc')


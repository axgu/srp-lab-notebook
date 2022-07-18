#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import scipy as scp
import sklearn
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

get_ipython().run_line_magic('store', '-r logperformAcc')
get_ipython().run_line_magic('store', '-r upper_acc')
get_ipython().run_line_magic('store', '-r acc')


# In[2]:


# Compare accuracies
xAx = [i for i in range(0,90)]
plt.plot(xAx, logperformAcc, label="log-reg")
plt.plot(xAx, upper_acc, label="perm-test")
plt.plot(xAx, acc, label="lstm")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xlim(0,90)
plt.title("Time-varying Classification Accuracy")
plt.legend()
plt.show()


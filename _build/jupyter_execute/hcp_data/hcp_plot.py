#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import torch
import random
from matplotlib import pyplot as plt

from lstm_data_prep import prep
from attention import test_model, initialize_encoder_decoder
from lstm import test_model_lstm, initialize_lstm

get_ipython().run_line_magic('store', '-r logperformAcc')
get_ipython().run_line_magic('store', '-r lstm_accuracy')
get_ipython().run_line_magic('store', '-r attention_accuracy')


# In[2]:


# Compare accuracies
xAx = [i for i in range(0,90)]
plt.plot(xAx, logperformAcc, label="log-reg")
plt.plot(xAx, attention_accuracy, label="attention-seq")
plt.plot(xAx, lstm_accuracy, label="lstm")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xlim(0,90)
plt.title("Time-varying Classification Accuracy")
plt.legend()
plt.show()


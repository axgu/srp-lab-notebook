#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import torch
import random
from matplotlib import pyplot as plt

from lstm_data_prep import prep
from rnn_perm_test import iterateSeq, test_random_labels, test_random_column_labels, test_random_features
from attention import test_model, initialize_encoder_decoder


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"
with open('HCP_movie_watching.pkl','rb') as f:
    TS = pickle.load(f)

input_size = 300
hidden_size = 32
n_layers = 1
seq_len = 90
class_num = 15

_, test_loader = prep(TS)


# In[3]:


encoder, attent, decoder, encoder_optimizer, decoder_optimizer, loss_fn = initialize_encoder_decoder(input_size, hidden_size, class_num)

check = torch.load("encoder-decoder.pt")
encoder.load_state_dict(check["encoder"])
decoder.load_state_dict(check["decoder"])

encoder.eval()
decoder.eval()

attention_accuracy, loss_val = test_model(encoder, decoder, test_loader, seq_len, loss_fn)


# In[9]:


test_permutation_acc = iterateSeq(encoder, decoder, loss_fn, TS, numSamples = 3)
random_feature_acc = test_random_features(encoder, decoder, loss_fn, TS, num_samples = 3)


# In[5]:


# Test with random labels
# random_labels_samples_accuracy, _ = test_random_labels(encoder, decoder, loss_fn, TS)


# In[6]:


random_batch_labels_accuracy, _ = test_random_column_labels(encoder, decoder, loss_fn, TS, num_samples = 1)


# In[10]:


# Compare accuracies
xAx = [i for i in range(0,90)]
plt.plot(xAx, attention_accuracy, label="attention-encoder-decoder")
# plt.plot(xAx, attention_permutation_accuracy, label="permutation")
"""
# Test with randomly generated label for each (batch, time step)
for i, plot in enumerate(random_labels_samples_accuracy):
    plot = np.insert(plot, 0, plot[0])
    plt.plot(xAx, plot, label="random-" + str(i+1))
"""

# Test with randomly generated label for each batch, same across time steps within batch
for i, plot in enumerate(random_batch_labels_accuracy):
    plt.plot(xAx, plot, label="random-labels-" + str(i+1))

# Test with permuted inputs
for i, plot in enumerate(test_permutation_acc):
    plt.plot(xAx, plot, label="permutation-" + str(i+1))

# Test with randomly generated inputs
for i, plot in enumerate(random_feature_acc):
    plt.plot(xAx, plot, label="random-inputs" + str(i+1))
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xlim(0,90)
plt.title("Time-varying Classification Accuracy")
plt.legend()
plt.show()


# In[8]:


get_ipython().run_line_magic('store', 'attention_accuracy')


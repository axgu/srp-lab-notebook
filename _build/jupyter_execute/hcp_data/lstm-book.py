#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from lstm_data_prep import prep, numpy_prep
from rnn_perm_test import iterateLSTM, create_permutation_3d
from lstm import test_model_lstm, initialize_lstm


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"
with open('HCP_movie_watching.pkl','rb') as f:
    TS = pickle.load(f)

input_size = 300
hidden_size = 32
seq_len = 90
class_num = 15

train_loader, test_loader = prep(TS)


# In[ ]:


lstmModel = initialize_lstm(input_size, hidden_size, class_num, seq_len)
train_loss = lstmModel.train(train_loader, 100)


# In[12]:


lstmModel = initialize_lstm(input_size, hidden_size, seq_len)

lstm = lstmModel.model
lstmLoss = lstmModel.loss_fn
lstmOptim = lstmModel.optim

check = torch.load("lstm-model.pt")
lstm.load_state_dict(check["lstm"])
lstmOptim.load_state_dict(check["lstm_optimizer"])

lstm.eval()

lstm_accuracy = lstmModel.test_lstm(test_loader)


# In[21]:


# Run permutation test on model
# permutation_lstm_accuracy = iterateLSTM(lstmModel, lstmLoss, TS, num_samples=1)


# In[16]:


def generate_random_features(X_lens, num_batches, num_seq = 90, num_features = 300, pad=-100.):
    X_random = []
    for i in range(num_batches):
        X_batch = np.random.normal(size=(X_lens[i], num_features))
        if X_lens[i] < num_seq:
            X_pad = np.array([[-100.]*300]*(num_seq - X_lens[i]))
            X_batch = np.append(X_batch, X_pad, axis=0)
        X_random.append(X_batch)
    X_random = np.array(X_random)
    return X_random

def test_random_features(model, dictionary, seq_len = 90, num_samples = 1, batch_size = 32):
    _, _, X, y = numpy_prep(dictionary)
    X_lens = find_lens(X)
    random_features_acc = []
    for i in range(num_samples):
        X_random = generate_random_features(X_lens, X.shape[0])
        X_random_data = TensorDataset(torch.from_numpy(X_random).float(), torch.from_numpy(y).float())
        X_random_loader = DataLoader(X_random_data, shuffle=True, batch_size=batch_size)
        sample_acc = model.test_lstm(X_random_loader)
        random_features_acc.append(sample_acc)
    return random_features_acc


# In[10]:


def find_lens(X):
    X_lens = []
    for batch in X:
        count = 0
        for time in batch:
            if time[0] == -100.:
                break
            else:
                count += 1
        X_lens.append(count)
    return X_lens


# In[5]:


get_ipython().run_line_magic('store', 'permutation_lstm_accuracy')


# In[22]:


# Compare accuracies
xAx = [i for i in range(0,90)]
plt.plot(xAx, lstm_accuracy, label="lstm")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xlim(0,90)
plt.title("Time-varying Classification Accuracy")
plt.legend()
plt.show()


# In[7]:


get_ipython().run_line_magic('store', 'lstm_accuracy')


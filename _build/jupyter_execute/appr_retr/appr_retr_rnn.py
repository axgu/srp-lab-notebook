#!/usr/bin/env python
# coding: utf-8

# # Moving Circles LSTM

# In[1]:


import numpy as np
import pickle
import time
from matplotlib import pyplot as plt
import gc
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"


# In[2]:


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, n_classes, num_layers = 1, drop = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.n_classes = n_classes
        
        super(LSTMNetwork, self).__init__()
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x, x_len, h):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        lstm_output, hidden = self.lstm(x, h)
    
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True, padding_value=0., total_length=self.seq_length)

        output = output.contiguous().view(-1, self.seq_length, self.hidden_size)
        output = self.dropout(output)
        output = self.fc(output)
        output = output.view(-1, self.seq_length, self.n_classes)
        
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device), torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))


# In[3]:


class Model:
    def __init__(self, lstm, loss_fn, optimizer, seq_len, n_output):
        self.model = lstm
        self.loss_fn = loss_fn
        self.optim = optimizer
        self.seq_len = seq_len
        self.n_output = n_output

    def train(self, train_loader, n_epochs):
        train_loss = []
        best_loss = 1e10

        self.model.train()
        for i in range(n_epochs):
            start = time.time()
            avg_loss = 0.
            for X, y in train_loader:
                loss = 0.
                curr_batch_size = X.shape[0]
                h = self.model.init_hidden(curr_batch_size)
                X, y = X.to(device), y.to(device)
                self.optim.zero_grad()

                X_lens = self.find_lens(X)
                output, h = self.model(X, X_lens, h)

                output = torch.transpose(output, 1, 2)
                y = torch.transpose(y, 1, 2)

                loss += self.loss_fn(output, y)
                
                loss.backward()
                self.optim.step()
                avg_loss += loss.item()

            end = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start, end)
            if best_loss > avg_loss:
                best_loss = avg_loss
                self.saveModel()
            train_loss.append(avg_loss)

            print("Epoch " + str(i + 1) + "/" + str(n_epochs))
            print("Time: " + str(epoch_mins) + " minutes " + str(epoch_secs) + " seconds")
            print("Training loss: " + str(avg_loss))
            print()
        
        return train_loss

    def saveModel(self):
        torch.save({"lstm": self.model.state_dict(), "lstm_optimizer": self.optim.state_dict()}, 'lstm-model.pt')

    def epoch_time(self, start_time, end_time):
        elapsed = end_time - start_time
        elapsed_mins = int(elapsed / 60)
        elapsed_secs = int(elapsed - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def find_lens(self, X):
        X_lens = []
        for batch in X:
            count = 0
            for time in batch:
                if torch.count_nonzero(time) == 0:
                    break
                else:
                    count += 1
            X_lens.append(count)
        return X_lens

    def eval(self, test_loader):
        self.model.eval()
        test_loss = 0.
        correct = np.zeros(self.seq_len)
        total = np.zeros(self.seq_len)

        confusion = np.zeros((self.n_output, self.n_output))

        with torch.no_grad():
            for X, y in test_loader:
                curr_batch_size = X.shape[0]
                h = self.model.init_hidden(curr_batch_size)
                X, y = X.to(device), y.to(device)

                X_lens = self.find_lens(X)

                output, h = self.model(X, X_lens, h)
                
                top_value, top_index = output.topk(1)
                out = torch.zeros(curr_batch_size, self.seq_len, self.n_output)
                for clip in range(curr_batch_size):
                    for k in range(self.seq_len):
                        out[clip][k][top_index[clip][k].item()] = 1

                output = torch.transpose(output, 1, 2)
                y_hat = torch.transpose(y, 1, 2)
                test_loss += self.loss_fn(output, y_hat).item()
                
                correct, total = accuracy(y.cpu().numpy(), out.cpu().numpy(), correct, total)

                true, pred = devector(y.cpu().numpy(), out.cpu().numpy())

                confusion += confusion_matrix(true, pred)

        accArr = correct / total
        return accArr, confusion

    
    def random_samples(self, X, pad=0.):
        X_random = []
        X_lens = self.find_lens(X)
        for i in range(X.shape[0]):
            X_batch = np.random.normal(size=(X_lens[i], X.shape[-1]))
            if X_lens[i] < self.seq_len:
                X_pad = np.array([[pad]*X.shape[-1]]*(self.seq_len - X_lens[i]))
                X_batch = np.append(X_batch, X_pad, axis=0)
            X_random.append(X_batch)
        X_random = np.array(X_random)
        return X_random
    
    def rand_test(self, test_loader, n_samples=20, percentile=90):
        rand_acc_array = []
        for sample in range(n_samples):
            correct = np.zeros(self.seq_len)
            total = np.zeros(self.seq_len)
            
            self.model.eval()

            with torch.no_grad():

                for x, y in test_loader:
                    loss = 0.
                    curr_batch_size = x.shape[0]
                    X_random = torch.from_numpy(self.random_samples(x)).float().to(device)
                    h = self.model.init_hidden(curr_batch_size)
                    y = y.to(device)

                    X_lens = self.find_lens(x)
                    output, h = self.model(X_random, X_lens, h)
                    top_value, top_index = output.topk(1)
                    out = torch.zeros(curr_batch_size, self.seq_len, self.n_output)
                    for clip in range(curr_batch_size):
                        for k in range(self.seq_len):
                            out[clip][k][top_index[clip][k].item()] = 1

                    output = torch.transpose(output, 1, 2)
                    y_hat = torch.transpose(y, 1, 2)
                    loss += self.loss_fn(output, y_hat).item()

                    correct, total = accuracy(y.cpu().numpy(), out.cpu().numpy(), correct, total)
            lstm_rand_acc = correct / total
            
            rand_acc_array.append(lstm_rand_acc)
        plot = np.percentile(np.sort(np.array(rand_acc_array), axis=0), percentile, axis=0)
        return plot


# In[4]:


def vectorize(y, c=2):
    new_y = np.zeros((y.shape[0], y.shape[1], c))
    for batch in range(y.shape[0]):
        for time in range(y.shape[1]):
            cl = y[batch][time]
            if cl < c:
                new_y[batch][time][int(cl)] = 1.
    return new_y


# In[5]:


def findIndex(val, arr):
    index = -1
    for x in range(arr.size):
        if val == arr[x]:
            index = x
            break
    return index

def accuracy(target, prob, correct, tot):
    for i in range(prob.shape[1]):
        correctCount = correct[i]
        totalCount = tot[i]
        for j in range(prob.shape[0]):
            if np.count_nonzero(target[j][i]) != 0:
                if findIndex(1., target[j][i]) == findIndex(1., prob[j][i]):
                    correctCount += 1
                totalCount += 1
        correct[i] = correctCount
        tot[i] = totalCount
    return correct, tot


# In[6]:


def devector(true, pred):
    new_y_true = []
    new_y_pred = []
    for batch in range(true.shape[0]):
        for time in range(true.shape[1]):
            if np.count_nonzero(true[batch][time]) != 0:
                i = findIndex(1., true[batch][time])
                new_y_true.append(i)
                j = findIndex(1., pred[batch][time])
                new_y_pred.append(j)
    return np.array(new_y_true), np.array(new_y_pred)


# In[7]:


with open(f"appr_retr_timesegments.pkl", 'rb') as f:
    X_train, y_train, len_train, X_test, y_test, len_test = pickle.load(f)

for batch, batch_len in enumerate(len_train):
    y_train[batch, batch_len:] = np.array([2]*(y_train.shape[-1] - batch_len))

for batch, batch_len in enumerate(len_test):
    y_test[batch, batch_len:] = np.array([2]*(y_test.shape[-1] - batch_len))

y_train = vectorize(y_train)
y_test = vectorize(y_test)

batch_size = 32

train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# In[8]:


n_input = 300
n_hidden = 32
n_classes = 2
seq_len = 129
learning_rate = 5e-4
EPOCHS = 100

lstm = LSTMNetwork(n_input, n_hidden, seq_len, n_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)

model = Model(lstm, loss_fn, optimizer, seq_len, n_classes)


# In[9]:


train_loss = model.train(train_loader, EPOCHS)


# In[90]:


xAx = [i for i in range(1, EPOCHS+1)]
plt.plot(xAx, train_loss)
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.xlim(1, EPOCHS)
plt.xticks([j*5 for j in range(1, EPOCHS // 5 + 1)])
plt.title("Training Loss")
plt.show()


# In[12]:


lstm_accuracy, confusion = model.eval(test_loader)


# In[92]:


print(confusion)


# In[16]:


lstm_rand_acc = model.rand_test(test_loader, n_samples=20)


# In[17]:


xAx = [i for i in range(1,130)]
plt.plot(xAx, lstm_accuracy, label="lstm")
plt.plot(xAx, lstm_rand_acc, label="random")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xlim(0,130)
plt.title("Time-varying Classification Accuracy")
plt.legend()
plt.show()


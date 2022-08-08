import pickle
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim

# from lstm_data_prep import prep, setSeed
from eval_model import accuracy, epoch_time, find_lens

device = "cuda" if torch.cuda.is_available() else "cpu"

with open('HCP_movie_watching.pkl','rb') as f:
    TS = pickle.load(f)

input_size = 300
hidden_size = 32
seq_len = 90
class_num = 15

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_layers = 1, num_classes = 15, drop = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        super(LSTMNetwork, self).__init__()
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, x_len, h):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        lstm_output, hidden = self.lstm(x, h)
    
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True, padding_value=-100.)

        output = output.contiguous().view(-1, self.seq_length, self.hidden_size)
        output = self.dropout(output)
        output = self.fc(output)
        output = output.view(-1, self.seq_length, self.num_classes)
        
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device), torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

class Model:
    def __init__(self, lstm, loss_fn, optimizer, input_size, hidden_size, output_size, seq_len):
        self.model = lstm
        self.loss_fn = loss_fn
        self.optim = optimizer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
    

    def train(self, train_loader, epochs):
        train_loss = []
        best_loss = 1e10

        self.model.train()
        for i in range(epochs):
            start = time.time()
            avg_loss = 0.
            for X, y in train_loader:
                loss = 0.
                curr_batch_size = X.shape[0]
                h = self.model.init_hidden(curr_batch_size)
                X, y = X.to(device), y.to(device)
                self.optim.zero_grad()

                X_lens = find_lens(X)

                output, h = self.model(X, X_lens, h)

                output = torch.transpose(output, 1, 2)
                y = torch.transpose(y, 1, 2)

                loss += self.loss_fn(output, y)
                
                loss.backward()
                self.optim.step()
                avg_loss += loss.item()

            end = time.time()
            epoch_mins, epoch_secs = epoch_time(start, end)
            if best_loss > avg_loss:
                best_loss = avg_loss
                self.saveModel()
               
            print("Epoch " + str(i + 1) + "/" + str(epochs))
            print("Time: " + str(epoch_mins) + " minutes " + str(epoch_secs) + " seconds")
            print("Training loss: " + str(avg_loss))
            print()
            train_loss.append(avg_loss)
        
        return train_loss

    def saveModel(self):
        torch.save({"lstm": self.model.state_dict(), "lstm_optimizer": self.optim.state_dict()}, 'lstm-model.pt')

    def test_lstm(self, test_loader):
        self.model.eval()
        test_loss = 0.
        correct = np.zeros(self.seq_len)
        total = np.zeros(self.seq_len)

        with torch.no_grad():
            for X, y in test_loader:
                curr_batch_size = X.shape[0]
                h = self.model.init_hidden(curr_batch_size)
                X, y = X.to(device), y.to(device)

                X_lens = find_lens(X)

                output, h = self.model(X, X_lens, h)

                top_value, top_index = output.topk(1)
                out = torch.zeros(curr_batch_size, self.seq_len, 15)
                for clip in range(curr_batch_size):
                    for k in range(self.seq_len):
                        out[clip][k][top_index[clip][k].item()] = 1

                output = torch.transpose(output, 1, 2).to(device)
                y_hat = torch.transpose(y, 1, 2).to(device)

                test_loss += self.loss_fn(output, y_hat).item()
                
                correct, total = accuracy(y.cpu().numpy(), out.cpu().numpy(), correct, total)

        accArr = correct / total
        return accArr

def initialize_lstm(input_size, hidden_size, class_num, seq_len, lr = 0.001):
    lstm = LSTMNetwork(input_size, hidden_size, seq_len).to(device)
    lstm_optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model = Model(lstm, loss_fn, lstm_optimizer, input_size, hidden_size, class_num, seq_len)
    return model


"""
EPOCHS = 100
input_size = 300
hidden_size = 32
n_layers = 1
seq_len = 90
class_num = 15

train_loader, test_loader = prep(TS)
model, model_optimizer, loss_fn = initialize_lstm(input_size, hidden_size, seq_len)

train_loss = train_model(model, EPOCHS, train_loader, loss_fn, model_optimizer, seq_len)
accuracy_arr = test_model_lstm(model, test_loader, loss_fn)
print(accuracy_arr)
"""
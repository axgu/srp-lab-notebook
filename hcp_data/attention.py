import pickle
import time
import numpy as np
import torch
import random
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from lstm_data_prep import prep
from eval_model import accuracy, epoch_time, find_lens

device = "cuda" if torch.cuda.is_available() else "cpu"

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1, drop_prob=0):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, x, x_len, h):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(x, h)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=-100.)
        
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device), torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=1, drop_prob=0):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.attention = attention

        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.output_size, self.hidden_size, batch_first = True)
        self.classifier = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs, x_lens, batch_size):
        inputs = inputs.unsqueeze(1)
        lstm_out, hidden = self.lstm(inputs, hidden)

        # Calculate alignment scores
        alignment_scores = self.attention(lstm_out, encoder_outputs, batch_size)

        for b, batch in enumerate(alignment_scores):
            if x_lens[b] < 90:
                for i in range(90 - x_lens[b]):
                    alignment_scores[b][x_lens[b] + i] = 1e-10
        
        attn_weights = F.softmax(alignment_scores.view(batch_size, 1, -1), dim=2)

        context_vector = torch.bmm(attn_weights, encoder_outputs)

        output = torch.cat((lstm_out, context_vector), -1)
        output = self.classifier(output)
        return output, hidden, attn_weights

class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, batch_size):
        if self.method == "dot":
            return encoder_outputs.bmm(decoder_hidden.view(batch_size,-1,1)).squeeze(-1)
        
        elif self.method == "general":
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(batch_size,-1,1)).squeeze(-1)


def train_model(epochs, encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, seq_len, loss_fn, teacher_forcing_ratio = 0.5):
    epoch_loss = []
    best_loss = 1e10

    encoder.train()
    decoder.train()

    for epoch in range(epochs):

        start = time.time()
        avg_loss = 0.
        for batch, (X, y) in enumerate(train_loader):
            loss = 0.
            curr_batch_size = X.shape[0]
            h = encoder.init_hidden(curr_batch_size)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            inputs = X.to(device)
            X_lens = find_lens(inputs)
            encoder_outputs, h = encoder(inputs, X_lens, h)


            # Initial input is 0s
            decoder_input = torch.zeros(y.shape[0], y.shape[-1], device=device)
            decoder_hidden = h
            teacher_force = random.random() < teacher_forcing_ratio

            for j in range(0, seq_len):
                decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_outputs, X_lens, curr_batch_size)
                top_value, top_index = decoder_output.topk(1)

                
                new_input = torch.zeros(curr_batch_size, 15)
                for clip in range(curr_batch_size):
                    if j >= X_lens[clip]:
                        new_input = new_input
                    else:
                        new_input[clip][top_index[clip].item()] = 1
                
                if teacher_force:
                    decoder_input = torch.tensor(y[:, j, :], device=device)
                else:
                    decoder_input = torch.tensor(new_input, device=device)


                loss += loss_fn(decoder_output.view(curr_batch_size,-1), torch.tensor(y[:, j, :], device=device))

                if j == 0:
                    output = torch.tensor(new_input.unsqueeze(1))
                else:
                    output = torch.cat((output, new_input.unsqueeze(1)), 1)
            
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            avg_loss += loss.item() / (seq_len)

        epoch_loss.append(avg_loss)

        end = time.time()
        epoch_mins, epoch_secs = epoch_time(start, end)
        if best_loss > avg_loss:
            best_loss = avg_loss
            torch.save({"encoder":encoder.state_dict(),"decoder":decoder.state_dict(),"e_optimizer":encoder_optimizer.state_dict(),"d_optimizer":decoder_optimizer},"encoder-decoder.pt")

        print("Epoch " + str(epoch + 1) + "/" + str(epochs))
        print("Time: " + str(epoch_mins) + " minutes " + str(epoch_secs) + " seconds")
        print("Training loss: " + str(avg_loss))
        print()
     
    return epoch_loss
def test_model(encoder, decoder, test_loader, seq_len, loss_fn):
    encoder.eval()
    decoder.eval()

    test_loss = 0.
    correct = np.zeros(seq_len)
    total = np.zeros(seq_len)

    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            curr_batch_size = X.shape[0]
            h = encoder.init_hidden(curr_batch_size)

            inputs, labels = X.to(device), y.to(device)
            X_lens = find_lens(inputs)
            encoder_outputs, h = encoder(inputs, X_lens, h)

            decoder_input = torch.zeros(y.shape[0], y.shape[-1], device=device)
            decoder_hidden = h

            for j in range(0, seq_len):
                decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_outputs, X_lens, curr_batch_size)
                top_value, top_index = decoder_output.topk(1)
                
                new_input = torch.zeros(curr_batch_size, 15)
                for clip in range(curr_batch_size):
                    new_input[clip][top_index[clip]] = 1
                
                decoder_input = torch.tensor(new_input, device=device)

                test_loss += loss_fn(decoder_output.view(curr_batch_size,-1), torch.tensor(y[:, j, :], device=device)).item()

                if j == 0:
                    output = torch.tensor(new_input.unsqueeze(1))
                else:
                    output = torch.cat((output, new_input.unsqueeze(1)), 1)
            
            correct, total = accuracy(y.numpy(), output.numpy(), correct, total)
    test_loss /= len(test_loader)
    accArr = correct / total
    return accArr, test_loss

def initialize_encoder_decoder(input_size, hidden_size, class_num, lr = 0.01, device=device):
    encoder = EncoderLSTM(input_size, hidden_size).to(device)
    attent = Attention(hidden_size, "general")
    decoder = DecoderLSTM(hidden_size, class_num, attent).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    return encoder, attent, decoder, encoder_optimizer, decoder_optimizer, loss_fn

"""
with open('HCP_movie_watching.pkl','rb') as f:
    TS = pickle.load(f)

input_size = 300
hidden_size = 32
n_layers = 1
seq_len = 90
class_num = 15

EPOCHS = 23

train_loader, test_loader = prep(TS)
encoder, attent, decoder, encoder_optimizer, decoder_optimizer, loss_fn = initialize_encoder_decoder(input_size, hidden_size, class_num)

attention_loss = train_model(EPOCHS, encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, seq_len, loss_fn)
accuracy_arr, test_loss = test_model(encoder, decoder, test_loader, seq_len, loss_fn)
"""
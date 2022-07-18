import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_layers = 1, num_classes = 15):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        super(LSTMNetwork, self).__init__()
        self.lstm_linear_softmax = nn.Sequential(
            nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        
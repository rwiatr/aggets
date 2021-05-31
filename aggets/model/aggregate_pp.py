import torch
import torch.nn as nn
from torch.random import initial_seed
from aggets.model import simple
from aggets.model.aggregate import WindowConfig

class AutoregLstm(nn.Module):
    def __init__(self, input, output, in_len, out_len=1, num_layers=1, hidden=64):
        super(AutoregLstm, self).__init__()
        self.window_config = WindowConfig(output_sequence_length=out_len, input_sequence_length=in_len)
        self.input = input

        self.enc = nn.LSTM(input_size=hidden,
                           hidden_size=hidden,
                           num_layers=num_layers,
                           batch_first=True)
        self.dec = nn.LSTM(input_size=hidden,
                           hidden_size=hidden,
                           num_layers=num_layers,
                           batch_first=True)

        self.attention = nn.Linear(hidden, 1)
        self.softmax = nn.Softmax(dim=1)
        
        self.relu = nn.ReLU()

        self.output = output

    def encode(self, x):
        enc, _ = self.enc(x)
        return enc

    def decode(self, enc):
        for _ in range(self.window_config.output_sequence_length):
            
            energy = self.attention(enc)
            attention = self.softmax(energy)
            # context
            enc = torch.bmm(attention.permute(0,2,1), enc)

            enc, _ = self.dec(enc)
            yield enc

    def forward(self, x):
        x = self.input.forward(x)
        x = self.encode(x)
        x = list(self.decode(x))
        x = torch.cat(x, dim=1)
        return self.output.forward(x)


class Flatten(nn.Module):
    def __init__(self, out_seq):
        super(Flatten, self).__init__()
        self.reshape = Reshape(None)
        self.out_seq = out_seq

    def forward(self, x):
        # print(x.shape)
        self.reshape.shape = (-1, self.out_seq, *x.shape[2:])
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # print(x.shape)
        return x

    def reverse(self):
        return self.reshape


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)
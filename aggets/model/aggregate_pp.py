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

        self.attention = nn.Linear(hidden*2, 1)
        self.softmax = nn.Softmax(dim=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.output = output

    def encode(self, x):
        output, (hidden, cell) = self.enc(x)
        return output, hidden, cell

    def decode(self, enc_output, hidden, cell):
        for _ in range(self.window_config.output_sequence_length):
            seq_length = enc_output.shape[1]
            hidden_reshaped = hidden.repeat(seq_length, 1, 1)
            hidden_permuted = hidden_reshaped.permute(1, 0, 2)

            energy = self.tanh(self.attention(torch.cat((hidden_permuted, enc_output), dim=2)))
            attention = self.softmax(energy)
            context_vector = torch.bmm(attention.permute(0,2,1), enc_output)

            output, (hidden, cell) = self.dec(context_vector, (hidden, cell))
            yield output

    def forward(self, x):
        x = self.input.forward(x)
        output, hidden, cell = self.encode(x)
        x = list(self.decode(output, hidden, cell))
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

class FlatCatReverse(nn.Module):
    def __init__(self):
        super(FlatCatReverse, self).__init__()

    def forward(self, x):
        hist = x[:, :, :-self.lr_last]
        lr = x[:, :, -self.lr_last:]
        hist = hist.reshape(hist.shape[0], hist.shape[1], *self.hist_shape)

        return hist, lr

class FlatCat(nn.Module):
    def __init__(self):
        super(FlatCat, self).__init__()
        self.fcr = FlatCatReverse()

    def forward(self, x):
        hist, lr = x
        self.fcr.lr_last = lr.shape[-1]
        self.fcr.hist_shape = hist.shape[2:]
        hist = hist.reshape(hist.shape[0], hist.shape[1], -1)
        return torch.cat([hist, lr], dim=-1)

    def reverse(self):
        return self.fcr

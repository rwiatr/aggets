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

        self.enc = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True)
        
        self.dec = nn.LSTM(
            input_size=hidden*2,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True)

        self.fc_hidden = nn.Linear(hidden*2, hidden)
        self.fc_cell = nn.Linear(hidden*2, hidden)

        # hidden states from the encoder and hidden from prev_step in decoder
        # a(s[i-1], h[j])
        self.attn = nn.Linear(hidden*3, 1)

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.output = output

    def encode(self, x):
        # x shape (seq_len, N) N=batch size
        # enc of size (seq_len, N, num_directions * hidden_size) [hj]
        # (h_n, c_n) (num_layers * num_directions, N, hidden_size)
        enc, (hidden, cell) = self.enc(x)

        # backward and forward hidden (2, N, hidden_size)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return enc[:, -1:, ], hidden[:, -1:, ], cell[:, -1:, ]

    def decode(self, enc, hidden, cell):

        seq_length = enc.shape[0]


        energy = self.relu(self.attn(torch.cat((hidden, enc), dim=2)))
        attention = self.softmax(energy)

        ## elementwise multiply attention with encoder states
        #attention = attention.permute(1, 2, 0)
        #(N, 1, seq_length)
        #enc = enc.permute(1, 0, 2)
        #(N, seq_length, hidden_size*2)
        context_vec = torch.einsum("snk,snl->knl", attention, enc)
        # (N, 1, hidden_size*2) -> (1, N, hidden_size*2)

        enc, (hidden, cell) = self.dec(context_vec, (hidden, cell))
        return enc, hidden, cell

    def forward(self, x):
        x = self.input.forward(x)
        encoder_states, hidden, cell = self.encode(x)

        outputs = []

        # new decode
        for _ in range(self.window_config.output_sequence_length):
            output, hidden, cell = self.decode(encoder_states, hidden, cell)
            outputs += [output]
        

        x = torch.cat(outputs, dim=1)
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
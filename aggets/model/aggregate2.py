import torch
import torch.nn as nn
from aggets.model import simple
from aggets.model.aggregate import WindowConfig


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


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


class EncodeSingleOutput(nn.Module):
    def __init__(self, in_size, out_size):
        super(EncodeSingleOutput, self).__init__()
        self.fc = simple.mlp(features=in_size, num_layers=1, out_features=out_size)

    def forward(self, x):
        x = self.fc(x)
        return x


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

        self.output = output

    def decode(self, enc):
        for _ in range(self.window_config.output_sequence_length):
            enc, _ = self.dec(enc)
            yield enc

    def forward(self, x):
        x = self.input.forward(x)
        x = self.encode(x)
        x = list(self.decode(x))
        x = torch.cat(x, dim=1)
        return self.output.forward(x)

    def encode(self, x):
        enc, _ = self.enc(x)
        return enc[:, -1:, ]


class FAdapter(nn.Module):
    def __init__(self, learner):
        super(FAdapter, self).__init__()
        self.learner = learner

    def forward(self, X):
        ts, lr = X

        """
        ts = [batch, seq, feature]
        lerner:
        input = [batch, size, hist_id, hist_type, t_in]
        output = [batch * hist_id * hist_type, size, t_in + (pos, id, type)]
        """
        print(ts.shape)
        ts = self.learner(ts)
        print(ts.shape)
        return ts



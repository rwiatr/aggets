import torch
import torch.nn as nn
from torch.random import initial_seed
from aggets.model import simple
from aggets.model import fourier
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
        self.dec = nn.LSTM(input_size=hidden * 2,
                           hidden_size=hidden,
                           num_layers=num_layers,
                           batch_first=True)

        self.softmax = nn.Softmax(dim=1)
        # self.attention = AttentionLayer(hidden=hidden, in_len=in_len)
        self.attention = SlimAttentionLayer(hidden=hidden)
        # self.attention = FourierAttentionLayer(hidden=hidden, in_len=in_len)
        self.relu = nn.ReLU()

        self.output = output

    def encode(self, x):
        enc, hs = self.enc(x)
        # print(enc.shape, hs.shape)
        # enc = torch.Size([batch, ts_len, hidden])
        return enc, hs

    def decode(self, enc, hes):
        s = torch.zeros(enc.shape[0], 1, enc.shape[2], requires_grad=False)
        for step in range(self.window_config.output_sequence_length):
            c_s = self.attention(enc, s)
            s, hd = self.dec(c_s)
            yield s

    def forward(self, x):
        x = self.input.forward(x)
        x, hs = self.encode(x)
        x = list(self.decode(x, hs))
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


class AttentionLayer(nn.Module):
    def __init__(self, hidden=64, in_len=10, cat_s=True):
        super(AttentionLayer, self).__init__()
        self.hidden = hidden
        self.in_len = in_len
        self.cat_s = cat_s

        self.w = simple.mlp(hidden, num_layers=1, out_features=1, input_dropout=0.2, dropout=0.5)
        self.u = simple.mlp(hidden, num_layers=1, out_features=1, input_dropout=0.2, dropout=0.5)
        self.v = simple.mlp(in_len, num_layers=8, out_features=in_len, input_dropout=0.2, dropout=0.5)

    def forward(self, hs, s):
        # print('hs', hs.shape)
        # print('s', s.shape)
        whs = self.w(hs)
        # print('whs', whs.shape)
        us_t = torch.cat([self.u(s)] * self.in_len, dim=1)
        # print('us_t', us_t.shape)
        # print('us_t + whs', (us_t + whs).squeeze(-1).shape)
        e_t = self.v(nn.functional.tanh(us_t + whs).squeeze(-1))
        # print('e_t', e_t.shape)
        a_t = nn.functional.softmax(e_t, dim=1).unsqueeze(1)
        c_t = (a_t @ hs)

        if self.cat_s:
            # print('a_t, hs', a_t.shape, hs.shape)
            # print('c_t, s', c_t.shape, s.shape)
            return torch.cat([c_t, s], dim=-1)

        return c_t


class SlimAttentionLayer(nn.Module):
    def __init__(self, hidden=64, cat_s=True):
        super(SlimAttentionLayer, self).__init__()
        self.hidden = hidden
        self.cat_s = cat_s
        inner_hidden = hidden  # // 4

        self.w = simple.mlp(hidden, num_layers=4, out_features=inner_hidden, input_dropout=0.2, dropout=0.5)
        self.u = simple.mlp(hidden, num_layers=4, out_features=inner_hidden, input_dropout=0.2, dropout=0.5)
        self.v = simple.mlp(inner_hidden, num_layers=8, out_features=1, input_dropout=0.2, dropout=0.5)

    def forward(self, hs, s):
        '''
            hs torch.Size([32, 10, 256])
            s torch.Size([32, 1, 256])
            whs torch.Size([32, 10, 64])
            us_t torch.Size([32, 10, 64])
            us_t + whs torch.Size([32, 10, 64])
            e_t torch.Size([32, 10, 1])
        '''
        in_len = hs.shape[1]
        # print('hs', hs.shape)
        # print('s', s.shape)
        whs = self.w(hs)
        # print('whs', whs.shape)
        us_t = torch.cat([self.u(s)] * in_len, dim=1)
        # print('us_t', us_t.shape)
        us_t_whs = nn.functional.tanh(us_t + whs)
        # print('us_t + whs', us_t_whs.shape)
        e_t = self.v(us_t_whs)
        # print('e_t', e_t.shape)
        a_t = nn.functional.softmax(e_t, dim=1)
        # print('a_t', a_t.shape)
        c_t = a_t * hs
        # print('c_t', c_t.shape)
        c_t = c_t.sum(dim=1).unsqueeze(1)
        # print('c_t_sum', c_t.shape)

        if self.cat_s:
            # print('a_t, hs', a_t.shape, hs.shape)
            # print('c_t, s', c_t.shape, s.shape)
            return torch.cat([c_t, s], dim=-1)

        return c_t


class FourierAttentionLayer(nn.Module):

    def __init__(self, hidden=64, in_len=10, cat_s=True):
        super(FourierAttentionLayer, self).__init__()
        self.hidden = hidden
        self.in_len = in_len
        self.cat_s = cat_s
        inner_hidden = hidden
        self.w = simple.mlp(hidden, num_layers=4, out_features=inner_hidden, input_dropout=0.2, dropout=0.5)
        self.u = simple.mlp(hidden, num_layers=4, out_features=inner_hidden, input_dropout=0.2, dropout=0.5)
        self.v = simple.mlp(inner_hidden, num_layers=8, out_features=1, input_dropout=0.2, dropout=0.5)
        self.f = fourier.HistogramLerner(extra_dims=0, t_in=in_len)

    def forward(self, hs, s):
        '''
            hs torch.Size([32, 10, 256])
            s torch.Size([32, 1, 256])
            whs torch.Size([32, 10, 256])
            us_t torch.Size([32, 10, 256])
            us_t + whs torch.Size([32, 10, 256])
            e_t torch.Size([32, 10, 1])
            a_t torch.Size([32, 10, 1])
            c_t torch.Size([32, 10, 256])
            c_t_sum torch.Size([32, 1, 256])
            a_t, hs torch.Size([32, 10, 1]) torch.Size([32, 10, 256])
            c_t, s torch.Size([32, 1, 256]) torch.Size([32, 1, 256])
        '''
        in_len = hs.shape[1]
        # print('hs', hs.shape)
        # print('s', s.shape)
        whs = self.w(hs)
        # print('whs', whs.shape)
        us_t = torch.cat([self.u(s)] * in_len, dim=1)
        # print('us_t', us_t.shape)
        us_t_whs = nn.functional.tanh(us_t + whs)
        # print('us_t + whs', us_t_whs.shape)
        e_t = self.v(us_t_whs)
        # print('e_t', e_t.shape)
        a_t = nn.functional.softmax(e_t, dim=1)
        # print('a_t', a_t.shape)
        c_t = a_t * hs
        # print('c_t', c_t.shape)
        c_t = self.f(c_t)
        # print('c_t_sum', c_t.shape)

        if self.cat_s:
            # print('a_t, hs', a_t.shape, hs.shape)
            # print('c_t, s', c_t.shape, s.shape)
            return torch.cat([c_t, s], dim=-1)

        return c_t

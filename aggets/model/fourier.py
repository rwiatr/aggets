from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from aggets.model import simple

from aggets.model.aggregate import WindowConfig
import numpy as np

""" based on https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py"""


# Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


class Fop1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(Fop1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 1, normalized=True, onesided=True)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1) // 2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        # Return to physical space
        x = torch.irfft(out_ft, 1, normalized=True, onesided=True, signal_sizes=(x.size(-1),))
        return x


class FTWBlock(nn.Module):
    def __init__(self, modes, width):
        super(FTWBlock, self).__init__()
        self.modes = modes
        self.width = width
        self.ft = Fop1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.bn = torch.nn.BatchNorm1d(self.width)

    def forward(self, x):
        batch = x.shape[0]
        x0 = self.ft(x)
        x1 = self.w(x.view(batch, self.width, -1)).view(x.shape)
        x = self.bn(x0 + x1)
        return x


class HistogramBlock(nn.Module):
    def __init__(self, modes, width):
        super(HistogramBlock, self).__init__()
        self.modes = modes
        self.width = width
        self.ftw_seq = nn.Sequential(
            FTWBlock(modes, width),
            nn.ReLU(),
            FTWBlock(modes, width),
            nn.ReLU(),
            FTWBlock(modes, width),
            nn.ReLU(),
            FTWBlock(modes, width),
        )
        self.fc = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x shape is [batch, bins, t_in + 1]
        """
        x = x.permute(0, 2, 1)
        x = self.ftw_seq(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        # print(x.shape)
        return x


class HistogramEncoder:
    def __init__(self, size, hists, types, t_in):
        self.grid = torch.tensor(np.linspace(0, 2 * np.pi, size).reshape(1, size, 1), dtype=torch.float)
        self.hist_id = torch.tensor(np.linspace(0, 2 * np.pi, hists).reshape(hists, 1, 1), dtype=torch.float)
        self.type_id = torch.tensor(np.linspace(0, 2 * np.pi, types).reshape(1, 1, types), dtype=torch.float)
        self.size = size
        self.hists = hists
        self.types = types
        self.t_in = t_in

    def encode(self, x):
        """
        input = [batch, size, hist_id, hist_type, t_in]
        output = [batch * hist_id * hist_type, size, t_in + (pos, id, type)]
        """
        # flatten dims
        # print(x.shape)
        x = torch.cat([x[:, :, h] for h in range(x.shape[2])])
        x = torch.cat([x[:, :, h] for h in range(x.shape[2])])
        batch = x.shape[0]
        # print(x.shape, self.size, self.t_in)
        # add position information, histogram_id information and histogram type information
        x = torch.cat([x.reshape(batch, self.size, self.t_in), self.grid.repeat(batch, 1, 1)], dim=-1)
        x = torch.cat([x.reshape(batch, self.size, self.t_in + 1),
                       self.hist_id.repeat(batch // self.hists, self.size, 1)], dim=-1)
        x = torch.cat([x.reshape(batch, self.size, self.t_in + 2),
                       self.type_id.repeat_interleave(batch // self.types * self.size).reshape(batch, self.size, 1)],
                      dim=-1)
        return x

    def decode(self, x):
        step = x.shape[0] // self.types
        x = torch.stack([x[a * step:(a + 1) * step] for a in range(self.types)], dim=2)
        step = x.shape[0] // self.hists
        x = torch.stack([x[a * step:(a + 1) * step] for a in range(self.hists)], dim=2)
        return x[:, :, :, :, :self.t_in]

    @property
    def t_out(self):
        return self.t_in + 3


class MultiHistogramBlock(nn.Module):
    def __init__(self, modes, width, heads, t_in):
        super(MultiHistogramBlock, self).__init__()
        self.modes = modes
        self.width = width
        self.fc = nn.Linear(t_in, width)
        self.heads = nn.ModuleList([HistogramBlock(modes, width) for _ in range(heads)])

    def forward(self, x):
        x = self.fc(x)
        x = torch.sum(torch.stack([head(x) for head in self.heads]), dim=0)
        return x


class HistogramLerner(nn.Module):
    def __init__(self, size, hists, types, t_in):
        super(HistogramLerner, self).__init__()
        self.encoder = HistogramEncoder(size, hists, types, t_in)
        self.multi_block = MultiHistogramBlock(3, 64, 4, self.encoder.t_out)

    def forward(self, x):
        # x, _ = x
        # print('0) x =', x.shape)
        x = self.encoder.encode(x)
        # print('1) x =', x.shape)
        x = self.multi_block(x)
        # print('2) x =', x.shape)
        x = self.encoder.decode(x)
        # print('3) x =', x.shape)
        return x


class RecurrentLerner(nn.Module):
    def __init__(self, block, steps):
        super(RecurrentLerner, self).__init__()
        self.block = block
        self.steps = steps

    def forward(self, x):
        return torch.cat(list(self.repeat(x)), dim=-1)

    def repeat(self, x):
        for _ in range(self.steps):
            result = self.block(x)
            x = torch.cat([x[:, :, :, :, 1:], result], dim=-1)
            yield result


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
        # print(ts.shape)
        ts = torch.transpose(ts, 1, 2)
        ts = torch.unsqueeze(ts, 2)
        ts = torch.unsqueeze(ts, 2)
        ts = self.learner(ts)
        ts = torch.transpose(ts, 1, 2)
        ts = ts[:, :, :, 0, 0]
        # print(ts.shape)
        return ts


class FAdapter2(nn.Module):
    def __init__(self, learner):
        super(FAdapter2, self).__init__()
        self.learner = learner

    def forward(self, X):
        ts = X

        """
        ts = [batch, seq, hist_id, hist_type, features]
        lerner:
        input = [batch, size, hist_id, hist_type, t_in]
        output = [batch * hist_id * hist_type, size, t_in + (pos, id, type)]
        """
        # print(ts.shape)
        ts = torch.transpose(ts, 1, -1)
        ts = self.learner(ts)
        ts = torch.transpose(ts, 1, -1)
        # ts = ts[:, :, :, 0, 0]
        # print(ts.shape)
        return ts

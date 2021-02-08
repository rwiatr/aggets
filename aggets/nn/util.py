import torch
import torch.nn as nn


class SwapDims(nn.Module):
    def __init__(self, d0, d1):
        super(SwapDims, self).__init__()
        self.d0 = d0
        self.d1 = d1

    def forward(self, x):
        return torch.transpose(x, self.d0, self.d1)

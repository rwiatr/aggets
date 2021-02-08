import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, net):
        super(ResidualBlock, self).__init__()
        self.net = net

    def forward(self, X):
        delta = self.net(X)

        return X + delta

import torch.nn as nn
import numpy as np
from aggets.nn.util import SwapDims


def conv_1d(conv_width, features, hidden, out_features=1, mlp_layers=1):
    """
    Input: (batch, sequence, features)
    """
    conv_layers = nn.Sequential(
        *[SwapDims(1, 2),
          nn.Conv1d(in_channels=features, out_channels=hidden, kernel_size=(conv_width,)),
          SwapDims(1, 2)])
    mlp_layers = mlp(features=hidden, num_layers=mlp_layers, hidden=hidden, out_features=out_features)

    return nn.Sequential(conv_layers, mlp_layers)


def n_conv_1d(features, conv_layers=2, fc_layers=3, conv_width=8, pool_width=4, pool_stride=3,
              conv_out_feature_div=2, batch_norm=False, dropout=False, out_features=None, conv_pool=True,
              conv_relu=True,
              return_seq_reducer=False):
    """
    Input: (batch, sequence, features)
    """
    seq_reducer = []
    net_conv = []
    input_size = features
    rew_seq_inc = []

    for layer in range(conv_layers):
        new_input_size = int(input_size // conv_out_feature_div)
        c = nn.Conv1d(in_channels=input_size, out_channels=new_input_size, kernel_size=conv_width)
        seq_reducer.append(lambda s: s - (conv_width - 1))
        rew_seq_inc.append(lambda s: (s + (conv_width - 1)))
        net_conv.append(c)
        if conv_pool:
            m = nn.MaxPool1d(kernel_size=pool_width, stride=pool_stride)
            net_conv.append(m)
            seq_reducer.append(lambda s: np.ceil((s - (pool_width - 1)) / pool_stride))
            rew_seq_inc.append(lambda s: s + pool_width - 1)
        if batch_norm:
            net_conv.append(nn.BatchNorm1d(num_features=new_input_size))
        if conv_relu:
            net_conv.append(nn.ReLU())

        if dropout:
            net_conv.append(nn.Dropout())
        input_size = new_input_size

    net_fc = []
    for layer in range(fc_layers - 1):
        net_fc.append(nn.Linear(in_features=input_size, out_features=input_size))
        # net_fc.append(nn.BatchNorm1d(num_features=input_size))
        net_fc.append(nn.ReLU())
        if dropout:
            net_fc.append(nn.Dropout())

    net_fc.append(nn.Linear(in_features=input_size, out_features=out_features if out_features else input_size))

    conv_l = nn.Sequential(*net_conv)
    fc_l = nn.Sequential(*net_fc)

    net = nn.Sequential(*[
        SwapDims(1, 2),
        conv_l,
        SwapDims(1, 2),
        fc_l
    ])

    if not return_seq_reducer:
        return net
    else:
        rew_seq_inc.reverse()

        def increase_seq(s):
            for increase in rew_seq_inc:
                s = increase(s)
            return int(s)

        def reducer(s):
            for reduction in seq_reducer:
                s = reduction(s)
            return int(s)

        reducer.minimal_input_dim = increase_seq(1)
        return net, reducer


def lstm(features, num_layers=1, hidden=64, out_features=1):
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.out_features = out_features
            self.lstm = nn.LSTM(input_size=features, hidden_size=hidden,
                                num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden, out_features)

        def forward(self, x):
            res, _ = self.lstm(x)
            return self.fc(res)

    return LSTMModel()


def mlp(features, num_layers=1, hidden=64, out_features=1, batch_norm=False, dropout=False):
    class ActivationWrapper(nn.Module):
        def __init__(self, wrapped):
            super(ActivationWrapper, self).__init__()
            self.wrapped = wrapped
            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.wrapped(x)
            return self.activation(x)

    if num_layers == 1:
        return nn.Linear(features, out_features)

    layers = [nn.Linear(features, hidden), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout())

    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout())

    layers.append(nn.Linear(features, out_features))

    return nn.Sequential(*layers)


def classify(model):
    return nn.Sequential(model, nn.Sigmoid())

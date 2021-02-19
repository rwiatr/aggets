import torch.nn as nn
import torch
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
              conv_out_feature_div=2, batch_norm=True, dropout=True, out_features=None):
    """
    Input: (batch, sequence, features)
    """
    net_conv = []
    input_size = features
    for layer in range(conv_layers):
        new_input_size = int(input_size // conv_out_feature_div)
        c = nn.Conv1d(in_channels=input_size, out_channels=new_input_size, kernel_size=conv_width)
        m = nn.MaxPool1d(kernel_size=pool_width, stride=pool_stride)
        net_conv.append(c)
        net_conv.append(m)
        if batch_norm:
            net_conv.append(nn.BatchNorm1d(num_features=new_input_size))
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

    return nn.Sequential(*[
        SwapDims(1, 2),
        conv_l,
        SwapDims(1, 2),
        fc_l
    ])


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


def mlp(features, num_layers=1, hidden=64, out_features=1):
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
    return nn.Sequential(*(
            [ActivationWrapper(nn.Linear(features, hidden))] +
            [ActivationWrapper(nn.Linear(hidden, hidden)) for _ in range(num_layers - 2)] +
            [nn.Linear(hidden, out_features)]
    ))


def classify(model):
    return nn.Sequential(model, nn.Sigmoid())

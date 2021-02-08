import torch.nn as nn

from aggets.nn.util import SwapDims


def conv_1d(conv_width, features, hidden, out_features=1):
    """
    Input: (batch, sequence, features)
    """
    return nn.Sequential(*[
        SwapDims(1, 2),
        nn.Conv1d(in_channels=features, out_channels=hidden, kernel_size=(conv_width,)),
        SwapDims(1, 2),
        nn.ReLU(),
        nn.Linear(in_features=hidden, out_features=hidden),
        nn.ReLU(),
        nn.Linear(in_features=hidden, out_features=out_features)
    ])


def lstm(features, num_layers=1, hidden=64, out_features=1):
    class LSTM_model(nn.Module):
        def __init__(self):
            super(LSTM_model, self).__init__()
            self.out_features = out_features
            self.lstm = nn.LSTM(input_size=features, hidden_size=hidden,
                                num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden, out_features)

        def forward(self, x):
            res, _ = self.lstm(x)
            return self.fc(res)

    return LSTM_model()

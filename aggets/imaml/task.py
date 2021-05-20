from aggets import train
import torch.nn.functional as F
import torch.nn as nn

from aggets.model import simple


class Task:
    def __init__(self, name, window, patience, lrs,
                 task_size, model_size,
                 source='all', target='lr'):
        self.name = name
        self.window = window
        self.patience = patience
        self.lrs = lrs
        self.source = source
        self.target = target
        self.input = simple.mlp(features=task_size, num_layers=1, out_features=model_size)
        self.output = simple.mlp(features=model_size, num_layers=1, out_features=task_size)

    def loop(self, model):
        l_model = model.copy()

        train.train_window_models(self.wrap(l_model),
                                  self.window,
                                  patience=self.patience,
                                  validate=True,
                                  weight_decay=0,
                                  max_epochs=1000,
                                  lrs=self.lrs,
                                  source=self.source,
                                  target=self.target,
                                  log=False,
                                  criterion=lambda y0, y1: F.mse_loss(y0, y1) + F.mse_loss(l_model, model))

    def wrap(self, model):
        return nn.Sequential(self.input, model, self.output)

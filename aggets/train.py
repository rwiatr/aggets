import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class FitLoop:
    def __init__(self, stop, criterion, net, optimizer, log_every=100):
        self.stop = stop
        self.criterion = criterion
        self.net = net
        self.optimizer = optimizer
        self.log_every = log_every

    def fit(self, train_loader, validation_loader=None):
        if validation_loader is None:
            validation_loader = train_loader

        batch_num = 0
        val_losses = self.validate(validation_loader)
        while not self.stop.is_stop():
            train_losses = []

            for batch_id, (X, y) in enumerate(train_loader()):
                outputs = self.net(X)
                self.optimizer.zero_grad()

                loss = self.criterion(outputs, y)

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

                if ((batch_num + 1) % self.log_every) == 0:
                    print('{}epoch {} batch {} loss={:.3}, '
                          'MTL={:.3}, '
                          'MVL={:.3}'
                          '\t\t\t\t\r'
                          .format('*' if self.stop.is_best() else '',
                                  self.stop.epoch,
                                  batch_num + 1,
                                  loss.item(),
                                  np.mean(np.abs(train_losses)),
                                  np.mean(np.abs(val_losses))))
                batch_num += 1

            val_losses = self.validate(validation_loader)

            self.net.train()
            self.stop.update_epoch_loss(validation_loss=np.mean(np.abs(val_losses)),
                                        train_loss=np.mean(np.abs(train_losses)))

            if self.stop.is_best():
                self.stop.handler.save(mtype='best')

        self.stop.handler.save(mtype='last')
        self.net.eval()

    def validate(self, validation_loader):
        val_losses = []
        self.net.eval()
        with torch.no_grad():
            for batch_id, (X, y) in enumerate(validation_loader()):
                outputs = self.net(X)
                loss = self.criterion(outputs, y)
                val_losses.append(loss.item())
        return val_losses


class EarlyStop:

    def __init__(self, handler, patience=100, max_epochs=None):
        self.patience = patience
        self.best_loss = np.inf
        self.failures = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.max_epochs = max_epochs if max_epochs is not None else np.inf
        self.is_best_loss = False
        self.handler = handler

    def update_epoch_loss(self, train_loss, validation_loss):
        self.is_best_loss = self.handler.update_epoch_loss(validation_loss)
        self.train_losses.append(train_loss)
        self.val_losses.append(validation_loss)
        self.epoch += 1
        if self.is_best_loss:
            self.failures = 0
            self.best_loss = validation_loss
        else:
            self.failures += 1

    def is_best(self):
        return self.is_best_loss

    def is_stop(self):
        return (self.failures > self.patience) or (self.epoch >= self.max_epochs)

    def plot_loss(self, plot_train_loss=False, moving_avg=100):
        if moving_avg:
            mvn_avg = len(self.val_losses) // moving_avg
            plt.plot(pd.Series(self.val_losses).rolling(max(mvn_avg, 1), center=True).mean(), label='Validation')
        else:
            plt.plot(self.val_losses, label='Validation')
        if plot_train_loss:
            if moving_avg:
                mvn_avg = len(self.val_losses) // moving_avg
                plt.plot(pd.Series(self.train_losses).rolling(max(mvn_avg, 1), center=True).mean(), label='Train')
            else:
                plt.plot(self.train_losses, label='Train')
        plt.title('Loss during training')
        plt.xlabel('epoch')
        plt.legend()


class ModelHandler:

    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf
        self.success_updates = 0

    def best(self, path=None):
        path = path if path else self.path
        self.model.load_state_dict(torch.load(path + '.best.bin'))

    def last(self, path=None):
        path = path if path else self.path
        self.model.load_state_dict(torch.load(path + '.last.bin'))

    def save(self, path=None, mtype='last'):
        path = path if path else self.path
        torch.save(self.model.state_dict(), path + mtype + '.bin')

    def update_epoch_loss(self, loss):
        if loss >= self.best_loss:
            return False
        else:
            self.best_loss = loss
            self.success_updates += 1
            return True


def train_window_model(model, window, lr=0.001, criterion=nn.MSELoss(), plot_loss=True, model_handler=None,
                       max_epochs=100, patience=1000, log_every=1000, weight_decay=0, path='model.bin'):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    handler = model_handler if model_handler else ModelHandler(model=model, path=path)
    stop = EarlyStop(patience=patience, max_epochs=max_epochs, handler=handler)
    loop = FitLoop(
        stop=stop,
        net=model,
        criterion=criterion,
        optimizer=optimizer,
        log_every=log_every
    )

    loop.fit(lambda: window.train, lambda: window.val)
    if plot_loss:
        stop.plot_loss(plot_train_loss=True)

    return handler


def train_model(model, data, lr=0.001, criterion=nn.MSELoss(), plot_loss=True, model_handler=None,
                max_epochs=100, patience=1000, log_every=1000, weight_decay=0, path='model.bin'):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    handler = model_handler if model_handler else ModelHandler(model=model, path=path)
    stop = EarlyStop(patience=patience, max_epochs=max_epochs, handler=handler)
    loop = FitLoop(
        stop=stop,
        net=handler.model,
        criterion=criterion,
        optimizer=optimizer,
        log_every=log_every
    )

    loop.fit(lambda: data.train, lambda: data.val)
    if plot_loss:
        stop.plot_loss(plot_train_loss=True)

    return handler

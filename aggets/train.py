import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class FitLoop:
    def __init__(self, stop, criterion, net, optimizer, log_every=100, path=None):
        self.stop = stop
        self.criterion = criterion
        self.net = net
        self.optimizer = optimizer
        self.log_every = log_every
        self.path = path

    def fit(self, train_loader, validation_loader=None, load_best=True):
        if validation_loader is None:
            validation_loader = train_loader

        batch_num = 0
        val_losses = None
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
                    print('epoch {} batch {} loss={:.3}, '
                          'MTL={:.3}, '
                          'MVL={:.3}'
                          '\t\t\t\t\r'
                          .format(self.stop.epoch,
                                  batch_num + 1,
                                  loss.item(),
                                  np.mean(np.abs(train_losses)),
                                  np.mean(np.abs(val_losses)) if val_losses is not None else np.nan))
                batch_num += 1

            val_losses = []
            self.net.eval()
            with torch.no_grad():
                for batch_id, (X, y) in enumerate(validation_loader()):
                    outputs = self.net(X)
                    loss = self.criterion(outputs, y)
                    val_losses.append(loss.item())

            self.net.train()
            self.stop.update_epoch_loss(validation_loss=np.mean(np.abs(val_losses)),
                                        train_loss=np.mean(np.abs(train_losses)))

            if self.stop.is_best() and self.path is not None:
                torch.save(self.net.state_dict(), self.path)
        if self.path is not None and load_best:
            self.load_best_state()

    def load_best_state(self):
        self.net.load_state_dict(torch.load(self.path))
        self.net.eval()


class EarlyStop:

    def __init__(self, patience=100, max_epochs=None):
        self.patience = patience
        self.best_loss = np.inf
        self.failures = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.max_epochs = max_epochs if max_epochs is not None else np.inf
        self.is_best_loss = False

    def update_epoch_loss(self, train_loss, validation_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(validation_loss)
        self.epoch += 1

        if validation_loss < self.best_loss:
            self.failures += 1
            self.is_best_loss = False
        else:
            self.failures = 0
            self.best_loss = validation_loss
            self.is_best_loss = True

    def is_best(self):
        return self.is_best_loss

    def is_stop(self):
        return (self.failures > self.patience) or (self.epoch >= self.max_epochs)

    def plot_loss(self, plot_train_loss=False):
        plt.plot(self.val_losses)
        if plot_train_loss:
            plt.plot(self.train_losses)
        plt.show()


def train_window_model(model, window, lr=0.001, criterion=nn.MSELoss(), plot_loss=True,
                       max_epochs=100, patience=1000, log_every=1000):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    stop = EarlyStop(patience=patience, max_epochs=max_epochs)
    loop = FitLoop(
        stop=stop,
        net=model,
        criterion=criterion,
        optimizer=optimizer,
        log_every=log_every,
        path='model.bin'
    )

    loop.fit(lambda: window.train, lambda: window.val, load_best=False)
    if plot_loss:
        stop.plot_loss(plot_train_loss=True)

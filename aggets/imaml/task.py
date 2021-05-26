import os

from torch.optim.adam import Adam

from aggets import train, util
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from aggets.model import simple
from aggets.model.aggregate import WindowConfig
from aggets.train import ModelHandler, ModelImprovementStop, FitLoop
from aggets.ds import data_catalog
import aggets.model.aggregate2 as agg_m
import copy
import pandas as pd
import matplotlib.pyplot as plt


class FLT(nn.Module):
    def __init__(self):
        super(FLT, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Rev(nn.Module):
    def __init__(self):
        super(Rev, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], 1, -1)


class Imaml:

    def __init__(self, tasks, model):
        self.tasks = tasks
        self.model = model
        self.outer_optimizer = optim.Adam(params=model.parameters(), lr=.001)

    def step(self):
        grad_list = []
        for task in self.tasks:
            task.set_model(model)
            grad_list.append(task.train())

        self.outer_optimizer.zero_grad()
        weight = torch.ones(len(grad_list))
        weight = weight / torch.sum(weight)
        grad = self.mix_grad(grad_list, weight)
        grad_log = self.apply_grad(self.model, grad)
        self.outer_optimizer.step()

    def plot(self, axs=None):
        for task in self.tasks:
            task.model_context.plot(axs=axs)
        for task in self.tasks:
            _, axs = plt.subplots(ncols=3, figsize=(20, 6))
            task.window.plot_lr(axs=axs, offsets=[0])
            task.window.plot_model(model=task.model_context.wrapped, axs=axs,
                                   other={'y_offset': -1, 'source': 'agg', 'target': 'lr'})

    def mix_grad(self, grad_list, weight_list):
        mixed_grad = []
        for g_list in zip(*grad_list):
            g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
            mixed_grad.append(torch.sum(g_list, dim=0))
        return mixed_grad

    def apply_grad(self, model, grad):
        '''
        assign gradient to model(nn.Module) instance. return the norm of gradient
        '''
        grad_norm = 0
        for p, g in zip(model.parameters(), grad):
            if p.grad is None:
                p.grad = g
            else:
                p.grad += g
            grad_norm += torch.sum(g ** 2)
        grad_norm = grad_norm ** (1 / 2)
        return grad_norm.item()


class ModelContext:

    def __init__(self, name, model, in_size, out_size, model_size, criterion=nn.functional.mse_loss, lr=0.001):
        self.model = copy.deepcopy(model)
        self.wrapped = nn.Sequential(FLT(),
                                     simple.mlp(in_size, out_features=model_size),
                                     self.model,
                                     simple.mlp(model_size, out_features=out_size),
                                     Rev())
        self.optimizer = Adam(lr=lr, params=self.wrapped.parameters())
        self.criterion = criterion
        self.train_losses = []
        self.reg_losses = []
        self.grad_losses = []
        self.epochs_per_train = 2
        self.max_parts = 20
        self.name = name
        self.wrapped.window_config = WindowConfig(1, 1, 0)
        self.wrapped.name = name

    def set_params(self, outer):
        for param_to, param_form in zip(self.model.parameters(), outer.parameters()):
            param_to.data.copy_(param_form)

    def train(self, train_set_fn):
        for epoch in range(self.epochs_per_train):
            epoch_loss = 0
            parts = 0
            for X, y in train_set_fn:
                self.optimizer.zero_grad()
                y_hat = self.wrapped(X)
                loss = self.criterion(y_hat, y)
                loss.backward()
                epoch_loss += loss.item()
                self.optimizer.step()

                parts += 1
                if parts == self.max_parts:
                    break
            self.train_losses.append(epoch_loss)

    def get_loss(self, data_set_fn):
        loss = torch.Tensor([0])
        for X, y in data_set_fn:
            y_hat = self.wrapped(X)
            loss += self.criterion(y_hat, y)
        return loss

    def pull_back(self, outer, lam=0.0):
        self.optimizer.zero_grad()
        loss = self.reg_loss(lam, outer)
        loss.backward()
        self.reg_losses.append(loss.item())
        self.optimizer.step()  # do we need this?

    def reg_loss(self, lam, outer):
        dist = 0
        for param_to, param_form in zip(self.model.parameters(), outer.parameters()):
            dist += torch.sum((param_form - param_to) ** 2)
        return 0.5 * lam * dist

    def cg(self, wrapped_window, lam):
        params = list(self.model.parameters())

        train_loss = self.get_loss(wrapped_window.train)
        val_loss = self.get_loss(wrapped_window.val)

        train_grad = torch.autograd.grad(train_loss, params, create_graph=True)
        # train_grad = torch.autograd.grad(train_loss, params)
        train_grad = nn.utils.parameters_to_vector(train_grad)
        val_grad = torch.autograd.grad(val_loss, params)
        val_grad = nn.utils.parameters_to_vector(val_grad)
        grad_loss = Cg(model, lam).cg(train_grad, val_grad, params)
        self.grad_losses.append(grad_loss)

        return grad_loss

    def plot(self, axs=None, normalize=True):
        if axs is None:
            fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

        long_index = np.arange(start=0, stop=len(self.train_losses), step=1)
        short_index = np.arange(start=0, stop=len(self.reg_losses), step=1)
        # df = pd.DataFrame(index=short_index, data=self.reg_losses, columns=[f'{name}-regloss'])
        # if normalize:
        #     df = (df - df.min()) / (df.max() - df.min())
        # df.plot(ax=axs[0])
        df = pd.DataFrame(index=long_index, data=self.train_losses, columns=[f'{self.name}-trainloss'])
        if normalize:
            df = (df - df.min()) / (df.max() - df.min())
        df.plot(ax=axs[1])
        # pd.DataFrame(index=short_index, data=self.grad_losses).plot()


class Task:
    def __init__(self, name, window, model,
                 input_size, output_size, model_size, lam=1,
                 source='all', target='lr', lr=0.001):
        self.name = name
        self.window = window
        self.source = source
        self.target = target
        self.lam = lam
        self.model = model
        self.model_context = ModelContext(model=model, in_size=input_size,
                                          out_size=output_size, model_size=model_size,
                                          lr=lr, name=name)

    def set_model(self, model):
        self.model_context.set_params(model)

    def train(self):
        wrapped_window = self.window.wrapped(self.model_context.wrapped.window_config,
                                             other={'y_offset': -1, 'source': 'agg', 'target': 'lr'})
        self.model_context.train(wrapped_window.train)
        self.model_context.pull_back(self.model, self.lam)

        return self.model_context.cg(wrapped_window, self.lam)


class Cg:
    def __init__(self, model, lam):
        self.model = model
        self.lam = lam
        self.n_cg = 5

    @torch.no_grad()
    def cg(self, in_grad, outer_grad, params):
        x = outer_grad.clone().detach()
        r = outer_grad.clone().detach() - self.hv_prod(in_grad, x, params)
        p = r.clone().detach()
        for i in range(self.n_cg):
            Ap = self.hv_prod(in_grad, p, params)
            alpha = (r @ r) / (p @ Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new @ r_new) / (r @ r)
            p = r_new + beta * p
            r = r_new.clone().detach()
        return self.vec_to_grad(x)

    def vec_to_grad(self, vec):
        pointer = 0
        res = []
        for param in self.model.parameters():
            num_param = param.numel()
            res.append(vec[pointer:pointer + num_param].view_as(param).data)
            pointer += num_param
        return res

    @torch.enable_grad()
    def hv_prod(self, in_grad, x, params):
        hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
        hv = torch.nn.utils.parameters_to_vector(hv).detach()
        # precondition with identity matrix
        return hv / self.lam + x


class HessianVal:
    def matrix_evaluator(self, task, lam, regu_coef=1.0, lam_damping=10.0, x=None, y=None):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """

        def evaluator(v):
            hvp = self.hessian_vector_product(task, v, x=x, y=y)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av

        return evaluator

    def hessian_vector_product(self, vector, model, tloss, x=None, y=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        # xt, yt = x, y
        # tloss = self.get_loss(xt, yt)
        grad_ft = torch.autograd.grad(tloss, model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        # vec = utils.to_device(vector, self.use_gpu)
        h = torch.sum(flat_grad * vector)
        hvp = torch.autograd.grad(h, model.parameters())
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])

        return hvp_flat


def validate(self):
    return 0


# class ImamlWindowTrainer:
#
#     def __init__(self, model, lr, weight_decay, criterion,
#                  max_improvement=np.inf, max_epochs=100, log_every=1000, log=True):
#         self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#         self.handler = ModelHandler(model=model, path=None)
#         self.stop = ModelImprovementStop(max_improvement=max_improvement, max_epochs=max_epochs, handler=self.handler)
#         self.loop = FitLoop(
#             stop=self.stop,
#             net=model,
#             criterion=criterion,
#             optimizer=self.optimizer,
#             log_every=log_every,
#             log=log
#         )
#
#     def train(self, window, plot_loss=True, validate=True, train_loss_mul=1, optimize=True, log=True):
#         self.handler.reset()
#         self.loop.fit(lambda: window.train, (lambda: window.val) if validate else None, optimize=optimize)
#         if plot_loss and log:
#             self.stop.plot_loss(plot_train_loss=True, train_loss_mul=train_loss_mul)
#
#         if max_improvement == np.inf


if __name__ == '__main__':
    binary = data_catalog.Binary()
    import aggets.ds.aggregate_nd as agg_nd


    def make_window(data, name, window_size=50):
        train_np = data['train']
        val_np = data['val']
        test_np = data['test']
        file_name = f'{name}-ws{window_size}.bin'
        hist_bins = 20
        hist_dim = 1
        if not os.path.exists(file_name):
            window = agg_nd.window_generator(train_np.to_numpy(), val_np.to_numpy(), test_np.to_numpy(),
                                             window_size=window_size, e=0.00001, hist_bins=hist_bins, hist_dim=hist_dim)
            window.init_structures()
            util.save(window, path=file_name)
        return util.load(path=file_name), train_np.shape[1] - 1, hist_bins, hist_dim


    data_types = {
        'agr_a': make_window(binary.agr_a(), 'agr_a', window_size=500),
        'agr_g': make_window(binary.agr_g(), 'agr_g', window_size=500),
        'sea_a': make_window(binary.sea_a(), 'sea_a', window_size=500),
        'sea_g': make_window(binary.sea_g(), 'sea_g', window_size=500),
        'hyper_f': make_window(binary.hyper_f(), 'hyper_f', window_size=500),
        # 'weather': make_window(binary.weather(), 'weather'),
        # 'electric': make_window(binary.electric(), 'electric')
    }

    tasks = []
    model_size = 256
    model = simple.mlp(features=model_size, num_layers=6, out_features=model_size)
    model.name = 'mlp'

    for data_type in data_types:
        data, dims, hist_bins, hist_dim = data_types[data_type]
        print(dims, hist_bins, hist_dim)
        tasks.append(Task(name=data_type, window=data, lr=0.001,
                          input_size=dims * hist_dim * hist_bins * 2, output_size=dims + 1,  # THIS IS WRONG FIX
                          model_size=model_size,
                          model=model))

    imaml = Imaml(tasks=tasks, model=model)
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

    for _ in range(50):
        imaml.step()
    imaml.plot(axs=axs)
    plt.show()

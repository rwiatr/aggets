from sklearn import metrics, linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def to_score(models, dfs, offset=0, rolling_frac=0.05):
    scores = []
    for model, df in zip(models, dfs[offset:]):
        X = df[:, :-1]
        y = df[:, -1]
        y_hat = model(X)
        scores.append(metrics.roc_auc_score(y, y_hat[:, 1]))
    if rolling_frac is None:
        return pd.DataFrame(index=range(offset, len(dfs)), data=scores)
    rolling = np.ceil(len(dfs) * rolling_frac)
    return pd.DataFrame(index=range(offset, len(dfs)), data=scores).rolling(int(rolling)).mean()


def to_dist(v0s, v1s, offset, rolling_frac=0.05):
    scores = []
    for v0, v1 in zip(v0s, v1s[offset:]):
        scores.append(torch.sqrt(torch.sum((v0 - v1) ** 2)))
    if rolling_frac is None:
        return pd.DataFrame(index=range(offset, len(v1s)), data=scores)
    rolling = np.ceil(len(v1s) * rolling_frac)
    return pd.DataFrame(index=range(offset, offset + len(scores)), data=scores).rolling(int(rolling)).mean()


def apply_to_dist(v0s, v1s, offset, dist_fn, rolling_frac=0.05, scale=1):
    scores = []
    for v0, v1 in zip(v0s, v1s):
        assert v0.shape == v1.shape, f'{v0.shape} != {v1.shape}'
        scores.append(dist_fn(v0, v1) / scale)
    if rolling_frac is None:
        return pd.DataFrame(index=range(offset, len(v1s)), data=scores)
    rolling = np.ceil(len(v1s) * rolling_frac)
    return pd.DataFrame(index=range(offset, offset + len(scores)), data=scores).rolling(int(rolling)).mean()


def to_lr(coefs):
    lr = linear_model.LogisticRegression().fit(np.random.rand(2, coefs.shape[0] - 1), [0, 1])
    lr.coef_ = coefs[:-1].reshape(1, -1).cpu().detach().numpy()
    lr.intercept_ = coefs[-1].cpu().detach().numpy()
    return lr


class PlotLrsVsData:

    def __init__(self, train_lrs, val_lrs, test_lrs, train_df, val_df, test_df):
        self.train_lrs = train_lrs
        self.val_lrs = val_lrs
        self.test_lrs = test_lrs

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def plot(self, d_types=['train', 'val', 'test'], offsets=[0, 1, 20], axs=None, rolling_frac=0.05):
        for idx, d_type in enumerate(d_types):
            dfs, lrs = self._select_set(d_type)
            for offset in offsets:
                score = to_score(lrs, dfs, offset=offset, rolling_frac=rolling_frac)
                if axs is not None:
                    plt.sca(axs[idx])
                plt.plot(score, label=f'LR[offset={offset}]')
            if axs is not None:
                plt.sca(axs[idx])
            plt.title(f'AUC-{d_type}')
            plt.legend()

    def _select_set(self, d_type):
        lrs = self.train_lrs
        dfs = self.train_df
        if d_type == 'val':
            lrs = self.val_lrs
            dfs = self.val_df
        elif d_type == 'test':
            lrs = self.test_lrs
            dfs = self.test_df
        return dfs, [to_lr(lr).predict_proba for lr in lrs]


class PlotMetaModelVsData:

    def __init__(self, train_Xs, val_Xs, test_Xs, train_df, val_df, test_df, select):
        self.train_Xs, y = train_Xs
        self.val_Xs, y = val_Xs
        self.test_Xs, y = test_Xs

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.select = select


    def plot(self, model, offsets, d_types=['train', 'val', 'test'], axs=None, rolling_frac=0.05):
        for idx, d_type in enumerate(d_types):
            dfs, Xs = self._select_set(d_type)
            lrs = self._model_to_lrs(model, Xs)

            score = to_score(lrs, dfs[:-offsets[1]], offset=offsets[0], rolling_frac=rolling_frac)
            if axs is not None:
                plt.sca(axs[idx])
            plt.plot(score, label=f'{model.name}[input={offsets[0]}]')
            if axs is not None:
                plt.sca(axs[idx])
            plt.title(f'AUC-{d_type}')
            plt.legend()

    def _model_to_lrs(self, model, Xs):
        lrs = []
        meta = self.select(model(Xs))[:, 0]  # take only first model predicted
        for m in meta:
            lr = to_lr(m)
            lrs.append(lr.predict_proba)
        return lrs

    def _select_set(self, d_type):
        dfs = self.train_df
        Xs = self.train_Xs
        if d_type == 'val':
            Xs = self.val_Xs
            dfs = self.val_df
        elif d_type == 'test':
            Xs = self.test_Xs
            dfs = self.test_df
        return dfs, Xs


class PlotDist:

    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

    def plot(self, d_types=['train', 'val', 'test'], offsets=[1, 20], axs=None, rolling_frac=0.05):
        for idx, d_type in enumerate(d_types):
            dat = self._select_set(d_type)
            for offset in offsets:
                score = to_dist(dat, dat, offset=offset, rolling_frac=rolling_frac)
                if axs is not None:
                    plt.sca(axs[idx])
                plt.plot(score, label=f'LR[offset={offset}]')
            if axs is not None:
                plt.sca(axs[idx])
            plt.title(f'DIST-{d_type}')
            plt.legend()

    def _select_set(self, d_type):
        dat = self.train
        if d_type == 'val':
            dat = self.val
        elif d_type == 'test':
            dat = self.test
        return dat


class PlotMetaModelDist:

    def __init__(self, train_Xs, val_Xs, test_Xs, select):
        self.train_Xs_in, self.train_Xs_out = train_Xs
        self.val_Xs_in, self.val_Xs_out = val_Xs
        self.test_Xs_in, self.test_Xs_out = test_Xs
        self.select = select

    def plot(self, model, offsets, dist_fn=None, d_types=['train', 'val', 'test'], axs=None, rolling_frac=0.05):
        if dist_fn == None:
            dist_fn = lambda v0, v1: torch.sqrt(torch.sum((v0 - v1) ** 2))
        for idx, d_type in enumerate(d_types):
            _in, out = self._select_set(d_type)
            _eval = model(_in)
            # scale down by the output length
            score = apply_to_dist(self.select(_eval), self.select(out), dist_fn=dist_fn,
                                  offset=offsets[0], rolling_frac=rolling_frac, scale=offsets[1])
            if axs is not None:
                plt.sca(axs[idx])
            plt.plot(score, label=f'{model.name}[input={offsets[0]}]')
            if axs is not None:
                plt.sca(axs[idx])
            plt.title(f'DIST-{d_type}')
            plt.legend()

    def _select_set(self, d_type):
        _in = self.train_Xs_in
        out = self.train_Xs_out
        if d_type == 'val':
            _in = self.val_Xs_in
            out = self.val_Xs_out
        elif d_type == 'test':
            _in = self.test_Xs_in
            out = self.test_Xs_out
        return _in, out

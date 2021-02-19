import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import metrics, linear_model
import aggets.agge as agge
import pickle

"""
based on https://www.tensorflow.org/tutorials/structured_data/time_series
"""


def save(wg, path='window.bin'):
    with open(path, 'wb') as handle:
        pickle.dump(wg, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(path='window.bin'):
    with open('filename.pickle', 'rb') as handle:
        wg = pickle.load(handle)
    return wg


class WindowGenerator:
    def __init__(self, df_train, window_size, window_stride, label_columns,
                 slide_percent=0.5, train_proportion=0.5,
                 exclude_labels=True, sampling=32, sample_frac=1 / 16,
                 sub_windows=1000):
        self.window_size = window_size
        self.window_stride = window_stride
        self.slide_percent = slide_percent
        self.label_columns = label_columns
        self.train_proportion = train_proportion
        self.exclude_labels = exclude_labels
        self.sampling = sampling
        self.df_train = df_train
        self.df_test = df_test
        self.sample_frac = sample_frac

        self.sub_windows = sub_windows

        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}
        self.agges = None
        self.lr = None

    def init_structures(self):
        self.agges = self.init_aggregates()
        self.lr = self.init_models()

    def init_models(self):
        value_columns = [column for column in self.df.columns if column not in self.label_columns]
        lrs = []
        for chunk in np.array_split(self.df, self.sub_windows):
            lr = []
            for attempt in range(self.sampling):
                sample = chunk.sample(frac=self.sample_frac, replace=True)
                if len(lrs) == 0 and len(lr) == 0:
                    print(f"training {self.sub_windows} windows with {self.sampling}"
                          f" regressions with sample size {sample.shape[0]}")
                fit = linear_model.LinearRegression().fit(sample[value_columns], sample[self.label_columns])
                lr_vec = np.zeros(fit.coef_.shape[1] + 1)
                lr_vec[:-1] = fit.coef_.reshape(-1)
                lr_vec[-1] = fit.intercept_
                lr.append(torch.Tensor(lr_vec))
            lrs.append(torch.stack(lr))

        """ (window, attempt, lr_vec) """
        return torch.stack(lrs)

    def init_aggregates(self):
        column_ordering = {column: self.df[column].unique() for column in self.value_columns}

        agges = []
        for chunk in np.array_split(self.df, self.sub_windows):

            attempts = []
            for attempt in range(self.sampling):
                chunk = chunk.sample(frac=self.sample_frac, replace=True)
                if chunk.shape[0] == 0:
                    continue
                encoder = agge.AggregateEncoder()
                encoder.fit(chunk[value_columns], chunk[self.label_columns])
                models = encoder.model_vector(column_ordering=column_ordering)
                attempts.append(torch.Tensor(models))
            agges.append(torch.stack(attempts))

        """ (window, attempt, aggregate_vec) """
        return torch.stack(agges)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.window_size}',
            f'Input slide: {self.slide_percent * 100}%',
            f'Training set proportion: {self.train_proportion * 100}%',
            f'Label column name(s): {self.label_columns}',
            f'Window count: {self.window_count}'])

    @property
    def full_window_size(self):
        return (self.window_size + 1) * 3

    @property
    def window_count(self):
        return self.sub_windows // self.full_window_size

    def make_dataset(self, shuffle=True, batch_size=32):
        """
        aggs       = (window, attempt, aggregate_vec)
        source_lrs = (1, attempt, lr)
        target_lrs = (1, attempt, lr)
        """

        class BulkDataSet(data.Dataset):
            def __init__(self, aggs, source, target):
                self.aggs = torch.transpose(aggs, 0, 1)
                self.source = torch.transpose(source, 0, 1)
                self.target = torch.transpose(target, 0, 1)

            def __len__(self):
                return len(self.aggs)

            def __getitem__(self, idx):
                return (self.aggs[idx], self.source[idx]), self.target[idx]

        return data.DataLoader(BulkDataSet(self.agges[:-1]['aggs'],
                                           self.lr[:-1]['lr'],
                                           self.lr[1:]['lr']),
                               shuffle=shuffle,
                               batch_size=batch_size)

    @property
    def train(self):
        return self.make_dataset(self.train_agg)

    @property
    def val(self):
        return self.make_dataset(self.val_agg)

    @property
    def test(self):
        return self.make_dataset(self.test_agg)

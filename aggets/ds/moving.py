import torch
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import metrics

from aggets.ds.dataloader import DEFAULT_BATCH

"""
based on https://www.tensorflow.org/tutorials/structured_data/time_series
"""


class WindowGenerator:
    def __init__(self, df, window_size, label_columns,
                 slide_percent=0.5, train_proportion=0.5,
                 exclude_labels=True, sampling=None):
        self.window_size = window_size
        self.slide_percent = slide_percent
        self.label_columns = label_columns
        self.train_proportion = train_proportion
        self.exclude_labels = exclude_labels
        self.sampling = sampling
        self.df = df

        self.label_columns_indices = {name: i for i, name in
                                      enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(df.columns)}

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.window_size}',
            f'Input slide: {self.slide_percent * 100}%',
            f'Training set proportion: {self.train_proportion * 100}%',
            f'Label column name(s): {self.label_columns}',
            f'Window count: {self.window_count}'])

    def plot(self, model, moving_avg=100):
        train = []
        test = []
        val = []
        for window in self.windows:
            tst_auc, tr_auc, val_auc = window.auc(model)
            train.append(tst_auc)
            test.append(tr_auc)
            val.append(val_auc)

        plt.figure(figsize=(12, 8))
        for n, (data_set, label) in enumerate([(train, 'Train'), (test, 'Test'), (val, 'Validation')]):
            if moving_avg:
                mvn_avg = len(data_set) // moving_avg
                plt.plot(pd.Series(data_set).rolling(max(mvn_avg, 1), center=True).mean(), label='Validation')
            else:
                plt.plot(data_set, label=label)
        plt.legend()
        plt.title('AUC')
        plt.xlabel('window')
        plt.ylabel('AUC')

    def plots(self, models, moving_avg=100):
        train = []
        test = []
        val = []
        for model, window in zip(models, self.windows):
            tst_auc, tr_auc, val_auc = window.auc(model)
            train.append(tst_auc)
            test.append(tr_auc)
            val.append(val_auc)

        plt.figure(figsize=(12, 8))
        for n, (data_set, label) in enumerate([(train, 'Train'), (test, 'Test'), (val, 'Validation')]):
            if moving_avg:
                mvn_avg = len(data_set) // moving_avg
                plt.plot(pd.Series(data_set).rolling(max(mvn_avg, 1), center=True).mean(), label='Validation')
            else:
                plt.plot(data_set, label=label)
        plt.legend()
        plt.title('AUC')
        plt.xlabel('window')
        plt.ylabel('AUC')

    @property
    def window_count(self):
        return int((self.df.shape[0] - self.window_size) / (self.window_size * self.slide_percent))

    @property
    def windows(self):
        for window_id in range(self.window_count):
            start = int(window_id * self.window_size * self.slide_percent)
            end = start + self.window_size
            window_df = self.df[start:end]
            split0 = int(self.train_proportion * window_df.shape[0])
            split1 = int((window_df.shape[0] - split0) * 0.5) + split0
            yield DataSet(
                train_df=window_df[:split0],
                test_df=window_df[split0:split1],
                val_df=window_df[split1:],
                label_columns=self.label_columns,
                column_indices=self.column_indices,
                exclude_labels=self.exclude_labels,
                sampling=self.sampling
            )


class DataSet:
    def __init__(self, train_df, test_df, val_df, label_columns, column_indices, exclude_labels, sampling):
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.column_indices = column_indices
        self.label_columns = set(label_columns)
        self.exclude_labels = exclude_labels
        self.sampling = sampling

    def split(self, features):
        inputs = features
        labels = features
        if self.label_columns is not None:
            labels = torch.stack(
                [torch.Tensor(labels[:, self.column_indices[name]]) for name in self.label_columns],
                axis=-1
            )
            if self.exclude_labels:
                inputs = torch.stack(
                    [torch.Tensor(inputs[:, self.column_indices[name]]) for name in self.column_indices.keys()
                     if name not in self.label_columns],
                    axis=-1
                )
        return inputs, labels

    def auc(self, model):
        aucs = []
        for ds in [self.train, self.test, self.val]:
            preds_y = []
            true_y = []
            with torch.no_grad():
                for X, y_true in ds:
                    model.eval()
                    preds_y.append(model(X))
                    true_y.append(y_true)

                fpr, tpr, _ = metrics.roc_curve(torch.cat(true_y, dim=0), torch.cat(preds_y, dim=0))
                auc = metrics.auc(fpr, tpr)
                aucs.append(auc)
        return aucs

    def plot(self, model):
        plt.figure(figsize=(12, 8))
        for n, (data_set, title) in enumerate([(self.train_df, 'Train ROC Curve'),
                                               (self.test_df, 'Test ROC Curve'),
                                               (self.val_df, 'Val ROC Curve')]):
            plt.subplot(3, 1, n + 1)
            model.eval()
            with torch.no_grad():
                y_probas = model(data_set)
                skplt.metrics.plot_roc(y_probas, data_set[self.label_columns], title=title)

    def make_dataset(self, df, shuffle=True, batch_size=DEFAULT_BATCH):
        if self.sampling is not None:
            df = df.sample(frac=self.sampling)
        loader = data.DataLoader(df.to_numpy(), shuffle=shuffle, batch_size=batch_size)
        return map(lambda features: self.split(features), loader)

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

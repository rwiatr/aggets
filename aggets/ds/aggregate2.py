import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import aggets.agge as agge
from sklearn import linear_model
from sklearn import metrics

from aggets.model.aggregate import LrConv, LrNConv

"""
based on https://www.tensorflow.org/tutorials/structured_data/time_series
"""


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, label_columns,
                 chunk_size, samples=32, sample_frac=1.0,
                 label_stride=5, shuffle_ts=False):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_stride = label_stride
        self.shuffle_ts = shuffle_ts
        self.chunk_size = chunk_size
        self.samples = samples
        self.sample_frac = sample_frac
        # Work out the label column indices.
        self.label_columns = label_columns
        self.value_columns = [column for column in train_df.columns if column not in self.label_columns]
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        self.column_ordering = {column: train_df[column].unique() for column in self.value_columns}

        self.value_column_count = len(self.value_columns)
        self.unique_values = sum([len(val) for val in self.column_ordering.values()])
        self.aggregate_slice = slice(0, self.unique_values)
        self.source_slice = slice(self.unique_values, self.unique_values + self.value_column_count)
        self.target_slice = slice(self.unique_values + self.value_column_count,
                                  self.unique_values + 2 * self.value_column_count)

    def init_structures(self):
        self.train_agges = self.init_aggregates(self.train_df)
        self.train_lr = self.init_models(self.train_df)
        self.test_agges = self.init_aggregates(self.test_df)
        self.test_lr = self.init_models(self.test_df)
        self.val_agges = self.init_aggregates(self.val_df)
        self.val_lr = self.init_models(self.val_df)

    def init_models(self, df):
        lrs = []
        for chunk in self.chunks(df):
            lr = []
            for attempt in range(self.samples):
                sample = chunk.sample(frac=self.sample_frac, replace=True)
                fit = linear_model.LinearRegression().fit(sample[self.value_columns], sample[self.label_columns])
                lr_vec = np.zeros(fit.coef_.shape[1] + 1)
                lr_vec[:-1] = fit.coef_.reshape(-1)
                lr_vec[-1] = fit.intercept_
                lr.append(torch.Tensor(lr_vec))
            lrs.append(torch.stack(lr))

        """ (window, attempt, lr_vec) """
        return torch.stack(lrs)

    def init_aggregates(self, df):
        agges = []
        for chunk in self.chunks(df):
            attempts = []
            for attempt in range(self.samples):
                chunk = chunk.sample(frac=self.sample_frac, replace=True)
                if chunk.shape[0] == 0:
                    continue
                encoder = agge.AggregateEncoder()
                encoder.fit(chunk[self.value_columns], chunk[self.label_columns])
                models = encoder.model_vector(column_ordering=self.column_ordering)
                attempts.append(torch.Tensor(models))
            agges.append(torch.stack(attempts))

        """ (window, attempt, aggregate_vec) """
        return torch.stack(agges)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        aggregates = features[:, :, self.aggregate_slice]
        source = features[:, :, self.source_slice]
        target = features[:, self.label_stride:, self.target_slice]
        # print(aggregates.shape, source.shape, target.shape)
        return (aggregates, source), target

    def make_dataset(self, agg, lrs):
        # agg = torch.transpose(agg, 0, 1)
        # lrs = torch.transpose(lrs, 0, 1)
        """
        agg        = (window, attempt, aggregate_vec)
        lrs        = (window, attempt, lr)
        """
        agg = torch.transpose(agg, 0, 1)
        lrs = torch.transpose(lrs, 0, 1)
        """
        agg        = (attempt/batch, window/seq, aggregate_vec)
        lrs        = (attempt/batch, window/seq, lr)
        """

        agg = agg[:, :-1, :]
        source_lrs = lrs[:, :-1, :]
        target_lrs = lrs[:, 1:, :]

        class NpDataset(data.Dataset):
            def __init__(self, array): self.array = array

            def __len__(self): return len(self.array)

            def __getitem__(self, i): return self.array[i]

        def ts_dataset(features, stride, sequence_length, shuffle, batch_size):
            ts_data = []
            agg, source, target = features
            size = agg.shape[1]  # sequenc size
            # torch.Size([5, 139, 102])
            # print(agg.shape)
            # print(size, sequence_length)
            for k in np.arange(0, size - sequence_length, step=stride):
                for i in range(agg.shape[0]):
                    a = agg[i, k:(k + sequence_length)].reshape(sequence_length, -1)
                    s = source[i, k:(k + sequence_length)].reshape(sequence_length, -1)
                    t = target[i, k:(k + sequence_length)].reshape(sequence_length, -1)
                    ts_data.append(torch.cat([a, s, t], dim=1))

            dataset = NpDataset(ts_data)

            ds = data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
            return map(self.split_window, ds)

        return ts_dataset(features=(agg, source_lrs, target_lrs),
                          stride=1,
                          sequence_length=self.total_window_size,
                          shuffle=True,
                          batch_size=32)

    @property
    def train(self):
        return self.make_dataset(self.train_agges, self.train_lr)

    @property
    def val(self):
        return self.make_dataset(self.val_agges, self.val_agges)

    @property
    def test(self):
        return self.make_dataset(self.test_agges, self.test_lr)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def chunks(self, df, skip=0):
        chunks = df.shape[0] // self.chunk_size
        for chunk_id in range(skip, chunks):
            yield df[chunk_id * self.chunk_size:(chunk_id + 1) * self.chunk_size]

    def plot(self, model_data_fn, model=None, model_resid=False, model_name=None,
             set_type='test',
             lr_t0=True, lr_t1=True, lr0=True, last_train=True, no_lr=False,
             rolling=100):
        window = self
        lr = linear_model.LogisticRegression().fit(self.train_df[self.value_columns][:2], [0, 1])

        def init_regression(weights):
            lr.coef_ = weights[:-1].reshape(1, -1).detach().numpy()
            lr.intercept_ = weights[-1].detach().numpy()

        def measure_lr(data_set):
            lr_auc = []
            idx = []
            for k, lr_params, df in data_set:
                lr.init_regression(lr_params)
                y_hat = lr.predict_proba(df[window.value_columns])
                lr_auc.append(metrics.roc_auc_score(df[window.label_columns], y_hat[:, 1]))
                idx.append(k)

            return pd.DataFrame(index=idx, data=lr_auc).rolling(rolling).mean()

        def zip_enum(x, df, skip_chunks=0):
            return zip(range(skip_chunks, len(x)), x, self.chunks(df, skip=skip_chunks))

        def repeat(item, times):
            return [item for _ in range(times)]

        lr.init_regression = init_regression
        lr.measure = measure_lr

        sets = {'train': (self.train_lr[:, 0, :], self.train_df, self.train_agges[:, 0, :], self.train_lr),
                'val': (self.val_lr[:, 0, :], self.val_df, self.val_agges[:, 0, :], self.val_lr),
                'test': (self.test_lr[:, 0, :], self.test_df, self.test_agges[:, 0, :], self.test_lr)}
        lrs, df, agg, lrs_full = sets[set_type]

        if not no_lr:
            if lr_t0:
                x = lr.measure(zip_enum(lrs, df))  # LR(t) -> DF[t]
                plt.plot(x, label='LR[t]')
            if lr_t1:
                x = lr.measure(zip_enum(lrs, df, skip_chunks=1))  # LR(t) -> DF[t+1]
                plt.plot(x, label='LR[t+1]')
            if lr0:
                x = lr.measure(zip_enum(repeat(lrs[0], len(lrs)), df))  # LR(0) -> DF[t]
                plt.plot(x, label='LR[0]')
            if set_type != 'train' and last_train:
                x = lr.measure(zip_enum(repeat(self.train_lr[-1, 0, :], len(lrs)), df))  # last LR_train -> DF[t]
                plt.plot(x, label='LR[train]')

        def measure_model(data_set):
            model.eval()
            with torch.no_grad():
                model_aucs = []
                index = []
                for idx, dat, df in data_set:
                    lr_hat = model.forward(dat).detach().numpy().reshape(-1)
                    if model_resid:
                        lr_hat = lr_hat + dat[1][:, -1].detach().numpy().reshape(-1)
                    lr.coef_ = lr_hat[:-1].reshape(1, -1)
                    lr.intercept_ = lr_hat[-1]
                    y_hat = lr.predict_proba(df[window.value_columns])
                    model_aucs.append(metrics.roc_auc_score(df[window.label_columns], y_hat[:, 1]))
                    index.append(idx)

                return pd.DataFrame(index=index, data=model_aucs).rolling(rolling).mean()

        if model is not None:
            if model_data_fn is None:
                if isinstance(model, LrConv):
                    def prepare_model_input(data_set_lr, data_set_agg):
                        for k in range(len(data_set_agg) - (model.conv_width + 1)):
                            aggregates = data_set_agg[k:k + model.conv_width, :]
                            lr_v = data_set_lr[k:k + model.conv_width, :]
                            # target = data_set_lr[k + model.conv_width, :]
                            yield aggregates.reshape(1, *aggregates.shape), lr_v.reshape(1, *lr_v.shape)

                    model_data_fn = prepare_model_input

                if isinstance(model, LrNConv):
                    def prepare_model_input(data_set_lr, data_set_agg):
                        for k in range(len(data_set_agg) - (model.conv_width + 1)):
                            aggregates = data_set_agg[k:k + model.conv_width, :]
                            lr_v = data_set_lr[k:k + model.conv_width, :]
                            # target = data_set_lr[k + model.conv_width, :]
                            yield aggregates.reshape(1, *aggregates.shape), lr_v.reshape(1, *lr_v.shape)

                    model_data_fn = prepare_model_input

            model_input = list(model_data_fn(lrs, agg))
            x = measure_model(zip_enum(model_input, df, skip_chunks=model.conv_width))
            plt.plot(x, label='model' if model_name is None else model_name)

        plt.xlabel('window')
        plt.ylabel('AUC')
        plt.legend()

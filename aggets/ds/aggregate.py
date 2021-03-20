import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aggets.agge as agge
from sklearn import linear_model
from sklearn import metrics
import torch.nn.functional as F

"""
based on https://www.tensorflow.org/tutorials/structured_data/time_series
"""
import multiprocessing as mp


def make_aggregate(X, y, column_ordering, sample_frac, bin_count):
    X = X.sample(frac=sample_frac, replace=True)
    if X.shape[0] == 0:
        return None
    encoder = agge.AggregateEncoder()
    encoder.fit(X, y, bin_count=bin_count, normalize=False)
    encoded = encoder.model_vectors(column_ordering=column_ordering)
    return encoded


class WindowGenerator:
    def __init__(self, train_df, val_df, test_df,
                 label_columns, chunk_size,
                 input_sequence_length,
                 output_sequence_length=1,
                 df=None, samples=32, sample_frac=1.0, shuffle_reg=False, bin_count=1,
                 discretization=10, shuffle_input=False, shuffle_output=False, reverse_train=False,
                 double_target=True, density=True, one_row_ts=True, debug_output=False, mark_type=False,
                 train_histograms=False):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.shuffle_input = shuffle_input
        self.shuffle_output = shuffle_output
        self.reverse_train = reverse_train
        self.discretization = discretization
        self.density = density
        self.density_encoded = density
        self.one_row_ts = one_row_ts
        self.double_target = double_target
        self.debug_output = debug_output
        self.mark_type = mark_type
        self.train_histograms = train_histograms

        self.shuffle_reg = shuffle_reg
        self.chunk_size = chunk_size
        self.samples = samples
        self.sample_frac = sample_frac
        self.bin_count = bin_count
        # Work out the label column indices.
        self.label_columns = label_columns
        self.value_columns = [column for column in train_df.columns if column not in self.label_columns]
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        d_df = self.discretize(df if df is not None else train_df)
        self.column_ordering = {column: d_df[column].unique() for column in self.value_columns}

        self.value_column_count = len(self.value_columns)
        self.unique_values = sum([len(val) for val in self.column_ordering.values()])

    def init_structures(self):
        self.train_agges = self.init_aggregates(self.train_df)
        self.train_lr = self.init_models(self.train_df)
        self.test_agges = self.init_aggregates(self.test_df)
        self.test_lr = self.init_models(self.test_df)
        self.val_agges = self.init_aggregates(self.val_df)
        self.val_lr = self.init_models(self.val_df)

    def init_models(self, df):
        print('calculating models ...', end='\r')
        lrs = []
        chunks = list(self.chunks(df))
        for n, chunk in enumerate(chunks):
            lr = []
            for attempt in range(self.samples):
                sample = chunk.sample(frac=self.sample_frac, replace=True)
                fit = linear_model.LinearRegression().fit(sample[self.value_columns], sample[self.label_columns])
                lr_vec = np.zeros(fit.coef_.shape[1] + 1)
                lr_vec[:-1] = fit.coef_.reshape(-1)
                lr_vec[-1] = fit.intercept_
                lr.append(torch.Tensor(lr_vec))
            print(f'calculating models ... {n}/{len(chunks)}', end='\r')

            lrs.append(torch.stack(lr))

        print(f'                                                         ', end='\r')

        """ (window, attempt, lr_vec) """
        return torch.stack(lrs)

    def discretize(self, df):
        return (df[self.value_columns] * self.discretization).astype(int) / self.discretization

    def load_data(self, other):
        self.train_df = other.train_df
        self.val_df = other.val_df
        self.test_df = other.test_df
        # self.shuffle_input = other.shuffle_input
        # self.shuffle_output = other.shuffle_output
        # self.reverse_train = other.reverse_train
        self.discretization = other.discretization

        # self.shuffle_reg = other.shuffle_reg
        self.chunk_size = other.chunk_size
        self.samples = other.samples
        self.sample_frac = other.sample_frac
        self.bin_count = other.bin_count
        # Work out the label column indices.
        self.label_columns = other.label_columns
        self.value_columns = other.value_columns
        self.label_columns_indices = other.label_columns_indices
        self.column_indices = other.column_indices

        # self.input_sequence_length = other.input_sequence_length
        # self.output_sequence_length = other.output_sequence_length

        self.column_ordering = other.column_ordering

        self.value_column_count = other.value_column_count
        self.unique_values = other.unique_values
        self.train_agges = other.train_agges
        self.train_lr = other.train_lr
        self.test_agges = other.test_agges
        self.test_lr = other.test_lr
        self.val_agges = other.val_agges
        self.val_lr = other.val_lr

    def init_aggregates(self, df):
        print('calculating histograms ...', end='\r')
        threads = None
        threads = mp.cpu_count() - 1 if threads is None else threads
        with mp.Pool(processes=threads) as pool:
            chunks = []
            for n, chunk in enumerate(self.chunks(df)):
                y = chunk[self.label_columns]
                d_chunk = self.discretize(chunk)
                attempts = []
                for _ in range(self.samples):
                    attempt = pool.apply_async(
                        func=make_aggregate,
                        args=[d_chunk, y, self.column_ordering, self.sample_frac, self.bin_count]
                    )
                    attempts.append(attempt)
                chunks.append(attempts)
            print('calculating histograms ... executing futures ...', end='\r')
            b_chunks = []
            for n, b_attempts in enumerate(chunks):
                b_attempts = [attempt.get() for attempt in b_attempts]
                b_attempts = [torch.Tensor(tensor) for tensor, indices in b_attempts if attempt is not None]
                chunk = torch.stack(b_attempts)
                b_chunks.append(chunk)
                print(f'calculating histograms ... '
                      f'executing futures ... {(n + 1) * self.samples}/{(len(chunks) * self.samples)}',
                      end='\r')

        print(f'                                                                     ', end='\r')
        """ (window, attempt, aggregate_vec) """
        stack = torch.stack(b_chunks)
        return stack

    def __repr__(self):
        return '\n'.join([f'Label column name(s): {self.label_columns}'])

    def split_window(self,
                     lr_size,
                     agg_size,
                     in_len, out_len,
                     double_target,
                     return_density=False,
                     single_ts=True):
        def __split_window(features):
            """ features = [attempt, time, (p, d, lr), type]"""
            p_size = agg_size // 2
            source = features[:, :in_len, :, 0]
            target = features[:, -out_len:, :, 1]
            aggregates_p = source[:, :, :p_size]
            aggregates_d = source[:, :, p_size:agg_size]
            source = source[:, :, agg_size:agg_size + lr_size]

            if self.train_histograms:
                target = target[:, :, :p_size]
            else:
                target = target[:, :, agg_size:agg_size + lr_size]

            if self.debug_output:
                print(f'--- p {aggregates_p.shape}')
                for i in range(3):
                    print(aggregates_p[i, :, 0])
                print(f'--- d {aggregates_d.shape}')
                for i in range(3):
                    print(aggregates_d[i, :, 0])
                print(f'--- s {source.shape}')
                for i in range(3):
                    print(source[i, :, 0])
                print(f'--- t {target.shape}')
                for i in range(3):
                    print(target[i, :, 0])
            if return_density:
                if single_ts:
                    aggregates = torch.cat([aggregates_p, aggregates_d], dim=-1)
                else:
                    aggregates = torch.stack([aggregates_p, aggregates_d], dim=-2)
            else:
                aggregates = aggregates_p
            if self.debug_output:
                print(f'p={aggregates_p.shape}, d={aggregates_d.shape}, res={aggregates.shape}, '
                      f's={source.shape}, t={target.shape}')
            return (aggregates, source), target

        return __split_window

    def to_np_dataset(self, blob, in_len, out_len, with_reverse, step=1):
        class NpDataset(data.Dataset):
            def __init__(self, array): self.array = array

            def __len__(self): return len(self.array)

            def __getitem__(self, i): return self.array[i]

        data_len = max(in_len, out_len)
        out_start = min(in_len, out_len)
        buffer_size = in_len + out_len
        attempts = blob.shape[0]
        time = blob.shape[1]
        ts_data = []
        for k in np.arange(0, time - buffer_size, step=step):
            for a in range(attempts):
                """ blob = attempt, time, (p, d, lr)"""
                x = blob[a, k:k + data_len, :]
                y = blob[a, k + out_start:k + data_len + out_start, :]
                row = torch.stack([x, y], dim=-1)
                ts_data.append(row)
                if with_reverse:
                    ts_data.append(torch.flip(row, [0]))

        return NpDataset(ts_data)

    def to_dataset(self, agg, in_len, lrs, out_len, with_reverse):
        blob, a_size, lr_size = self.compress_into_single_matrix(agg, lrs)
        return self.to_np_dataset(blob=blob,
                                  in_len=in_len,
                                  out_len=out_len,
                                  with_reverse=with_reverse), a_size, lr_size

    def data_loader(self, agg, lrs, is_train=False, shuffle=True, batch_size=32):
        in_len = self.input_sequence_length
        out_len = self.output_sequence_length
        with_reverse = self.reverse_train and is_train
        dataset, a_size, lr_size = self.to_dataset(agg, in_len, lrs, out_len, with_reverse)

        dl = data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

        return map(self.split_window(
            return_density=self.density,
            single_ts=self.density,
            double_target=self.double_target,
            in_len=in_len, out_len=out_len,
            lr_size=lr_size,
            agg_size=a_size), dl)

    def compress_into_single_matrix(self, agg, lrs):
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
        if self.mark_type:  # TODO TEST IT
            ids_2 = torch.Tensor(np.linspace(0, agg.shape[2] - 1, agg.shape[2]))
            ids_2 = ids_2.repeat(1, agg.shape[0], agg.shape[1], 1).permute(1, 2, 3, 0)
            agg = torch.cat([ids_2, agg], dim=-1)

        if self.debug_output:
            ids_0 = torch.Tensor(np.linspace(0, agg.shape[0] - 1, agg.shape[0]))
            ids_1 = torch.Tensor(np.linspace(0, agg.shape[1] - 1, agg.shape[1]))
            ids_2 = torch.Tensor(np.linspace(0, agg.shape[2] - 1, agg.shape[2]))

            ids_0 = ids_0.repeat(1, agg.shape[1], agg.shape[2], 1).permute(3, 1, 2, 0)
            ids_1 = ids_1.repeat(1, agg.shape[0], agg.shape[2], 1).permute(1, 3, 2, 0)
            ids_2 = ids_2.repeat(1, agg.shape[0], agg.shape[1], 1).permute(1, 2, 3, 0)

            agg = torch.cat([ids_1, ids_0, ids_2, agg], dim=-1)

            ids_0 = torch.Tensor(np.linspace(0, lrs.shape[0] - 1, lrs.shape[0]))
            ids_1 = torch.Tensor(np.linspace(0, lrs.shape[1] - 1, lrs.shape[1]))

            ids_0 = ids_0.repeat(1, lrs.shape[1], 1).permute(2, 1, 0)
            ids_1 = ids_1.repeat(1, lrs.shape[0], 1).permute(1, 2, 0)
            lrs = torch.cat([ids_1, ids_0, lrs], dim=-1)
        agg = agg.reshape(agg.shape[0], agg.shape[1], -1)
        blob = torch.cat([agg, lrs], dim=-1)

        """
            blob = [attempt, window, data]
            data = [(ids)p, (ids)d, (ids)lr]
        """
        return blob, agg.shape[-1], lrs.shape[-1]

    def configure(self, window_config):
        self.input_sequence_length = window_config.input_sequence_length or self.input_sequence_length
        self.output_sequence_length = window_config.output_sequence_length or self.output_sequence_length
        print(f'config in={self.input_sequence_length}, out={self.output_sequence_length}')

    @property
    def train(self):
        return self.data_loader(self.train_agges, self.train_lr, is_train=True)

    @property
    def val(self):
        return self.data_loader(self.val_agges, self.val_lr)

    @property
    def test(self):
        return self.data_loader(self.test_agges, self.test_lr)

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

    def plot_feature_hist(self, feature_id, bins_per_hist, set_type='test', max_frames=None):
        agg, _, _ = self.retrieve_set(set_type, 1)
        if max_frames is not None:
            agg = agg[:max_frames]
        bucket = slice(bins_per_hist * feature_id, bins_per_hist * (feature_id + 1))

        names = [f'feature-{n}' for n in range(bins_per_hist)]
        bars = plt.bar(names, [1 for _ in range(bins_per_hist)])
        plt.xticks(range(len(names)), names, rotation=90)

        anim_frames = 10

        def animate(i):
            for n, b in enumerate(bars):
                k = i // anim_frames
                h0 = agg[k, bucket][n]
                h1 = agg[k + 1, bucket][n]
                m = (i % anim_frames + 1) / anim_frames
                b.set_height(h0 * (1 - m) + h1 * m)
            return b,

        return animation.FuncAnimation(plt.gcf(), animate, frames=(agg.shape[0] - 1) * anim_frames,
                                       interval=20, blit=True)

    def plot_hist_dist(self, model=None, set_type='test', loss=F.mse_loss, rolling=100, axs=None):
        ax_id = 0
        agg, lr = self.retrieve_aggregates(set_type, 1.0)
        dataset, _, _ = self.to_dataset(agg=agg, lrs=lr,
                                        in_len=self.input_sequence_length,
                                        out_len=self.output_sequence_length,
                                        with_reverse=False)
        split = self.split_window(lr_size=lr.shape[-1], agg_size=agg.shape[-1],
                                  in_len=self.input_sequence_length,
                                  out_len=self.output_sequence_length,
                                  return_density=self.density,
                                  single_ts=self.density,
                                  double_target=self.double_target)

        def measure_model(model, attempts):
            """ agg = [time, type, vals]"""
            result = np.zeros([attempts, len(dataset) // attempts])
            data = [] # attempt, idx
            for row in dataset:
                total_batch.append(row)

            for idx, row in enumerate(dataset):
                print(torch.stack([row], dim=0).shape)
                source, target = split(torch.stack([row], dim=0))
                result[idx % attempts, idx // attempts] = loss(model(source), target).item()

            return pd.DataFrame(data=result.mean(axis=0)).rolling(rolling).mean()

        # if tn is not None:
        #     tn = [tn] if not isinstance(tn, list) else tn
        #     for delta in tn:
        #         measure_tn(tn)
        if axs is not None:
            plt.sca(axs[ax_id])
            ax_id += 1

        if model is not None:
            models = [model] if not isinstance(model, list) else model

            for model in models:
                mloss = measure_model(model, self.train_lr.shape[0])
                plt.plot(mloss.rolling(rolling).mean(), label=model.name)

    def plot(self, model_data_fn, model=None, model_resid=False, model_name=None,
             set_type='test', plot_box_auc=False, box_mean_dist_from=None,
             lr_t0=True, lr_t1=True, lr0=True, last_train=True, no_lr=False, lr_tn=None,
             rolling=100, axs=None, train=1.0):
        window = self
        ax_id = 0
        if axs is not None:
            plt.sca(axs[ax_id])
            ax_id += 1

        lr = linear_model.LogisticRegression().fit(self.train_df[self.value_columns][:2], [0, 1])
        result_auc = {}

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

            return pd.DataFrame(index=idx, data=lr_auc)

        def zip_enum(x, df, skip_chunks=0):
            return zip(range(skip_chunks, len(x)), x, self.chunks(df, skip=skip_chunks))

        def repeat(item, times):
            return [item for _ in range(times)]

        lr.init_regression = init_regression
        lr.measure = measure_lr
        agg, df, lrs = self.retrieve_set(set_type, train)

        if not no_lr:
            if lr_t0:
                x = lr.measure(zip_enum(lrs, df))  # LR(t) -> DF[t]
                result_auc['LR[t]'] = x
                plt.plot(x.rolling(rolling).mean(), label='LR[t]')
            if lr_t1:
                x = lr.measure(zip_enum(lrs, df, skip_chunks=1))  # LR(t) -> DF[t+1]
                result_auc['LR[t+1]'] = x
                plt.plot(x.rolling(rolling).mean(), label='LR[t+1]')
            if lr_tn is not None:
                x = lr.measure(zip_enum(lrs, df, skip_chunks=lr_tn))  # LR(t) -> DF[t+1]
                result_auc[f'LR[t+{lr_tn}]'] = x
                plt.plot(x.rolling(rolling).mean(), label=f'LR[t+{lr_tn}]')
            if lr0:
                x = lr.measure(zip_enum(repeat(lrs[0], len(lrs)), df))  # LR(0) -> DF[t]
                result_auc['LR[0]'] = x
                plt.plot(x, label='LR[0]')
            if set_type != 'train' and last_train:
                x = lr.measure(zip_enum(repeat(self.train_lr[-1, 0, :], len(lrs)), df))  # last LR_train -> DF[t]
                result_auc['LR[train]'] = x
                plt.plot(x.rolling(rolling).mean(), label='LR[train]')

        def measure_model(data_set, model_resid):
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

                return pd.DataFrame(index=index, data=model_aucs)

        if model is not None:
            models = [model] if not isinstance(model, list) else model
            for idx, model in enumerate(models):
                if hasattr(model, 'name'):
                    name = model.name
                else:
                    name = f'model_{idx}'

                input_length = model.window_config.input_sequence_length or self.input_sequence_length
                if model_data_fn is None:
                    def prepare_model_input(data_set_lr, data_set_agg):
                        for k in range(len(data_set_agg) - (input_length + 1)):
                            aggregates = data_set_agg[k:k + input_length, :]
                            if self.one_row_ts and self.density:
                                aggregates = aggregates.reshape(aggregates.shape[0], -1)
                            elif not self.density and not self.density_encoded:
                                aggregates = aggregates
                            else:
                                raise Exception('illegal state')
                            lr_v = data_set_lr[k:k + input_length, :]

                            yield aggregates.reshape(1, *aggregates.shape), lr_v.reshape(1, *lr_v.shape)

                    model_data_fn = prepare_model_input

                model_input = list(model_data_fn(lrs, agg))
                x = measure_model(zip_enum(model_input, df, skip_chunks=input_length), False)
                result_auc[name] = x
                plt.plot(x.rolling(rolling).mean(), label=name if model_name is None else model_name)

                if model_resid:
                    x = measure_model(zip_enum(model_input, df, skip_chunks=input_length), True)
                    result_auc[f'{name}+resid'] = x
                    plt.plot(x.rolling(rolling).mean(), label=f'{name}+resid' if model_name is None else model_name)

        # plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(handles, labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xlabel('window')
        plt.ylabel('AUC')

        if plot_box_auc:
            if axs is not None:
                plt.sca(axs[ax_id])
                ax_id += 1
            names = []
            dfs = []
            for name, df in result_auc.items():
                dfs.append(df.to_numpy().reshape(-1))
                names.append(name)
            plt.boxplot(dfs)
            plt.xticks(range(1, len(names) + 1), names)

        if box_mean_dist_from is not None:
            if axs is not None:
                plt.sca(axs[ax_id])
                ax_id += 1
            names = []
            dfs = []
            for name, df in result_auc.items():
                if name == box_mean_dist_from:
                    continue
                errors = ((result_auc[box_mean_dist_from] - df).dropna()) ** 2
                dfs.append(errors.to_numpy().reshape(-1))
                names.append(name)
                print(f'MEAN {name} ERROR DISTANCE FROM {box_mean_dist_from} = '
                      f'{errors.to_numpy().mean()}')
            plt.boxplot(dfs)
            plt.xticks(range(1, len(names) + 1), names, rotation=90)

        return result_auc

    def retrieve_set(self, set_type, train):
        sets = {'train': (self.train_lr[:int(self.train_lr.shape[0] * train), 0, :],
                          self.train_df[:int(self.train_df.shape[0] * train)],
                          self.train_agges[:int(self.train_agges.shape[0] * train), 0, :, :],
                          self.train_lr[:int(self.train_lr.shape[0] * train)]),
                'val': (self.val_lr[:, 0, :], self.val_df, self.val_agges[:, 0, :], self.val_lr),
                'test': (self.test_lr[:, 0, :], self.test_df, self.test_agges[:, 0, :], self.test_lr)}

        lrs, df, agg, lrs_full = sets[set_type]

        return agg, df, lrs

    def retrieve_aggregates(self, set_type, train):
        if set_type == 'train':
            return self.train_agges[:int(self.train_agges.shape[0] * train), :, :, :], \
                   self.train_lr[:int(self.train_lr.shape[0] * train), :, :]
        if set_type == 'val':
            return self.val_agges[:, :, :, :], \
                   self.val_lr[:, :, :]
        if set_type == 'test':
            return self.test_agges[:, :, :, :], \
                   self.test_lr[:, :, :]

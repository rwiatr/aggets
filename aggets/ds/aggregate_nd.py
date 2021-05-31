from aggets.ds.aggregators import AggregatorWrapper, DdAggregator
from aggets.ds.dataloader import SimpleDataLoaderFactory
import numpy as np
import torch

from aggets.ds.lerners import LogisticLerner
from aggets.ds.plotters import PlotLrsVsData, PlotMetaModelVsData, PlotDist, PlotMetaModelDist
from aggets.util import cuda_if_possible


def as_np(df, val_cols, lab_col):
    return df[val_cols + [lab_col]].to_numpy()


def window_generator(train_np, val_np, test_np, hist_bins=5, hist_dim=2,
                     window_size=50, samples=5, sample_frac=0.8, e=0):
    maxs = np.max(np.stack([np.max(train_np, axis=0), np.max(val_np, axis=0), np.max(test_np, axis=0)]), axis=0)
    mins = np.min(np.stack([np.min(train_np, axis=0), np.min(val_np, axis=0), np.min(test_np, axis=0)]), axis=0)
    ranges = np.stack([mins - e, maxs + e]).transpose()

    print(f'ranges -> {ranges}')

    aggregator = DdAggregator(sample_frac, ranges=ranges, hist_bins=hist_bins, hist_dim=hist_dim)

    return WindowGenerator(
        train_np, val_np, test_np,
        window_size=window_size,
        aggregator=AggregatorWrapper(samples, aggregator, discretization=None),
        lerner=LogisticLerner(samples, sample_frac)
    )


class WindowGenerator:
    def __init__(self, train_np, val_np, test_np, aggregator, lerner, window_size,
                 dataloader_factory=SimpleDataLoaderFactory(),
                 device=cuda_if_possible()):
        self.window_size = window_size
        self.train_np_windows = list(self._to_windows(train_np))
        self.val_np_windows = list(self._to_windows(val_np))
        self.test_np_windows = list(self._to_windows(test_np))

        self.aggregator = aggregator
        self.lerner = lerner
        self.dataloader_factory = dataloader_factory
        self.device = device

    def init_structures(self):
        self.train_agges = self.aggregator.init_aggregates(self.train_np_windows).to(self.device)
        self.train_models = self.lerner.init_models(self.train_np_windows).to(self.device)
        self.test_agges = self.aggregator.init_aggregates(self.test_np_windows).to(self.device)
        self.test_models = self.lerner.init_models(self.test_np_windows).to(self.device)
        self.val_agges = self.aggregator.init_aggregates(self.val_np_windows).to(self.device)
        self.val_models = self.lerner.init_models(self.val_np_windows).to(self.device)

    def _to_windows(self, arr, skip=0):
        windows = arr.shape[0] // self.window_size
        for chunk_id in range(skip, windows):
            yield arr[chunk_id * self.window_size:(chunk_id + 1) * self.window_size]

    def train(self, in_len, out_len, other={}):
        return self.dataloader_factory.data_loader(self.train_agges, self.train_models, in_len, out_len, **other)

    def val(self, in_len, out_len, other={}):
        return self.dataloader_factory.data_loader(self.val_agges, self.val_models, in_len, out_len, **other)

    def test(self, in_len, out_len, other={}):
        return self.dataloader_factory.data_loader(self.test_agges, self.test_models, in_len, out_len, **other)

    def plot_lr(self, d_types=['train', 'val', 'test'], offsets=[0, 1, 20], axs=None, rolling_frac=0.05):
        return PlotLrsVsData(train_lrs=self.train_models[:, 0],
                             val_lrs=self.val_models[:, 0],
                             test_lrs=self.test_models[:, 0],
                             train_df=self.train_np_windows,
                             val_df=self.val_np_windows,
                             test_df=self.test_np_windows) \
            .plot(d_types, offsets, axs, rolling_frac)

    def plot_lr_dist(self, d_types=['train', 'val', 'test'], offsets=[1, 20, 50], axs=None, rolling_frac=0.05):
        return PlotDist(train=self.train_models[:, 0],
                        val=self.val_models[:, 0],
                        test=self.test_models[:, 0]) \
            .plot(d_types, offsets, axs, rolling_frac)

    def plot_agg_dist(self, d_types=['train', 'val', 'test'], offsets=[1, 20, 50], axs=None, rolling_frac=0.05,
                      select=lambda a: a[:, 0, 0]):  # default selector selects first sample and p histogram
        return PlotDist(train=select(self.train_agges),
                        val=select(self.val_agges),
                        test=select(self.test_agges)) \
            .plot(d_types, offsets, axs, rolling_frac)

    def plot_model(self, model, other={}, d_types=['train', 'val', 'test'], axs=None, rolling_frac=0.05,
                   select=lambda x: x):
        wrapped = self.wrapped(model.window_config, {**{'first_sample': True,
                                                        'shuffle': False,
                                                        'batch_size': self.train_agges.shape[0] * 5},
                                                     **other})

        return PlotMetaModelVsData(train_Xs=list(wrapped.train)[0],
                                   val_Xs=list(wrapped.val)[0],
                                   test_Xs=list(wrapped.test)[0],
                                   train_df=self.train_np_windows,
                                   val_df=self.val_np_windows,
                                   test_df=self.test_np_windows,
                                   select=select) \
            .plot(model=model,
                  offsets=(model.window_config.input_sequence_length, model.window_config.output_sequence_length),
                  d_types=d_types,
                  axs=axs,
                  rolling_frac=rolling_frac)

    def plot_model_agg_dist(self, model, other={}, d_types=['train', 'val', 'test'], axs=None, rolling_frac=0.05,
                            select=lambda k: k):
        wrapped = self.wrapped(model.window_config, {**{'first_sample': True,
                                                        'shuffle': False,
                                                        'batch_size': self.train_agges.shape[0] * 5,
                                                        'source': 'agg[0]',
                                                        'target': 'agg[0]'
                                                        }, **other})

        return PlotMetaModelDist(train_Xs=list(wrapped.train)[0],
                                 val_Xs=list(wrapped.val)[0],
                                 test_Xs=list(wrapped.test)[0],
                                 select=select) \
            .plot(model=model,
                  offsets=(model.window_config.input_sequence_length, model.window_config.output_sequence_length),
                  d_types=d_types,
                  axs=axs,
                  rolling_frac=rolling_frac)

    def wrapped(self, window_config=None, other={}):
        super_self = self

        class WrappedWindowGenerator:
            @property
            def train(self):
                return super_self.train(self.window_config.input_sequence_length,
                                        self.window_config.output_sequence_length,
                                        other)

            @property
            def val(self):
                return super_self.val(self.window_config.input_sequence_length,
                                      self.window_config.output_sequence_length,
                                      other)

            @property
            def test(self):
                return super_self.test(self.window_config.input_sequence_length,
                                       self.window_config.output_sequence_length,
                                       other)

            def configure(self, window_config):
                self.window_config = window_config
                return self

        return WrappedWindowGenerator().configure(window_config)

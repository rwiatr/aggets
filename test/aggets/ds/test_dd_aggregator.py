import numpy as np

from aggets.ds.aggregators import DdAggregator


def test_correct_dimensionality_0():
    dims = 10
    aggregator = DdAggregator(sample_frac=0.9, ranges=np.array([(0, 2) for _ in range(dims)]), hist_dim=2, hist_bins=10)
    windows_np = np.random.rand(100, dims) * 2
    aggregate = aggregator.aggregate(windows_np)

    """ 2 histogram types (prob and density), 36 all unique 2 element dimension pairs, dim0 10 bins, dim1 10 bins """
    assert aggregate.shape == (2, 36, 10, 10)


def test_correct_dimensionality_1():
    dims = 10
    aggregator = DdAggregator(sample_frac=0.9, ranges=np.array([(0, 2) for _ in range(dims)]), hist_dim=5, hist_bins=10)
    windows_np = np.random.rand(100, dims) * 2
    aggregate = aggregator.aggregate(windows_np)

    """ 2 histogram types (prob and density), 36 all unique 5 element dimension pairs, dim0 10 bins, dim1 10 bins,..."""
    assert aggregate.shape == (2, 126, 10, 10, 10, 10, 10)

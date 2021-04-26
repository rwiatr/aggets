import numpy as np

from aggets.ds.aggregators import DdAggregator, AggregatorWrapper


def test_correct_dimensionality_0():
    dims = 4
    aggregator = AggregatorWrapper(samples=5,
                                   aggregator=DdAggregator(sample_frac=0.9,
                                                           ranges=np.array([(0, 2) for _ in range(dims)]),
                                                           hist_dim=2, hist_bins=10),
                                   discretization=None)
    aggregate = aggregator.init_aggregates([(np.random.rand(100, dims) * 2), (np.random.rand(100, dims) * 2),
                                            (np.random.rand(100, dims) * 2), (np.random.rand(100, dims) * 2)])

    """ 4 windows, 5 samples, 2 types of histograms, 3 histogram combinations, 10 bins """
    assert aggregate.shape == (4, 5, 2, 3, 10, 10)

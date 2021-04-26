import numpy as np
import multiprocessing as mp
import torch

from aggets import agge


class BasicAggregator:

    def __init__(self, sample_frac, column_ordering, hist_bins=10):
        self.sample_frac = sample_frac
        self.column_ordering = column_ordering
        self.hist_bins = hist_bins

    def aggregate(self, X, y):
        sampling = np.random.randint(X.shape[0], size=int(X.shape[0] * self.sample_frac + 1))
        X = X[sampling, :]
        y = y[sampling, :]

        if X.shape[0] == 0:
            return None
        encoder = agge.AggregateEncoder()
        encoder.fit(X, y, bin_count=self.hist_bins, normalize=False)
        encoded = encoder.model_vectors(column_ordering=self.column_ordering)
        return encoded


class DdAggregator:

    def __init__(self, sample_frac, ranges, hist_dim=2, hist_bins=10):
        self.sample_frac = sample_frac
        self.ranges = ranges
        self.hist_dim = hist_dim
        self.hist_bins = hist_bins

    def _generate_dims(self, X_dims, hist_dim):
        for k in range(len(X_dims)):
            if hist_dim == 1:
                yield [X_dims[k]]
            else:
                for rest in self._generate_dims(X_dims[k + 1:], hist_dim - 1):
                    yield [X_dims[k]] + rest

    def aggregate(self, window_np):
        sampling = np.random.randint(window_np.shape[0], size=int(window_np.shape[0] * self.sample_frac + 1))
        window_np = window_np[sampling, :]
        X = window_np[:, :-1]
        y = window_np[:, -1]

        ps = []
        ds = []

        for dims in self._generate_dims(range(X.shape[1]), self.hist_dim):
            sub = X[:, dims]
            all, _ = np.histogramdd(sub, bins=self.hist_bins, range=self.ranges[dims])
            pos, _ = np.histogramdd(sub[y >= 1], bins=self.hist_bins, range=self.ranges[dims])
            den, _ = np.histogramdd(sub, bins=self.hist_bins, range=self.ranges[dims], density=True)
            p = pos / all
            np.nan_to_num(p, copy=False)
            ds.append(den)
            ps.append(p)

        """
            returns [2, unique_dim_combinations, hist_dim, hist_bins]
        """
        return np.stack([np.stack(ps), np.stack(ds)])


class AggregatorWrapper:
    def __init__(self, samples, aggregator, discretization):
        self.samples = samples
        self.aggregator = aggregator
        self.discretization = discretization

    def _discretize(self, arr):
        if self.discretization is None:
            return arr
        return (arr[:, 0:-1] * self.discretization).astype(int) / self.discretization

    def init_aggregates(self, windows_np):
        print('calculating histograms ...', end='\r')
        threads = 1
        threads = mp.cpu_count() - 1 if threads is None else threads
        with mp.Pool(processes=threads) as pool:
            windows = []
            for n, window_np in enumerate(windows_np):
                attempts = []
                window_np[:, :-1] = self._discretize(window_np[:, :-1])
                for _ in range(self.samples):
                    attempt = pool.apply_async(
                        func=self.aggregator.aggregate,
                        args=[window_np]
                    )
                    attempts.append(attempt)
                windows.append(attempts)
            print('calculating histograms ... executing futures ...', end='\r')
            b_chunks = []
            for n, b_attempts in enumerate(windows):
                b_attempts = [attempt.get() for attempt in b_attempts]
                # b_attempts = [torch.Tensor(tensor) for tensor, indices in b_attempts if attempt is not None]
                window = torch.stack([torch.Tensor(arr) for arr in b_attempts])
                b_chunks.append(window)
                print(f'calculating histograms ... '
                      f'executing futures ... {(n + 1) * self.samples}/{(len(windows) * self.samples)}',
                      end='\r')

        print(f'                                                                     ', end='\r')
        """ (window, attempt, aggregate_vec) """
        return torch.stack(b_chunks)

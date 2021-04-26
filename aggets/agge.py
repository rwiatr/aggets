import numpy as np
import pandas as pd

"""
Source: https://github.com/rwiatr/agge/
"""


class AggregateEncoder:

    def __init__(self):
        self.models = {}

    def fit(self, X, y, bin_count=1, normalize=True):
        self.bin_count = bin_count
        self.columns = X.columns
        for column in X.columns:
            feature_column = X[[column]]
            feature_column['y'] = y
            model = feature_column.groupby(column).agg(['sum', 'count'])
            model['p'] = model[('y', 'sum')] / model[('y', 'count')]
            model['d'] = model[('y', 'count')] / sum(model[('y', 'count')])
            model['bin'] = self._to_bins(bin_count, model[('y', 'count')])

            piv = model.pivot(columns=['bin'], values=[('p', ''), ('d', '')])

            zero_out = piv.isna()
            if normalize:
                piv = ((piv - piv.min()) / (piv.max() - piv.min())).fillna(1)
            piv[zero_out] = 0
            model[piv.columns] = piv
            self.models[column] = model[piv.columns]
        return self

    def model_vector(self, column_ordering=None, init_value=0):
        """ column_ordering maps a column name to an ordered set of column values """
        vectors = []
        columns = sorted(self.columns)
        for column in columns:
            if column_ordering is not None:
                sorted_values = column_ordering[column]
                flat = np.zeros(len(sorted_values) * self.bin_count)
                flat[:] = init_value
                for n, value in enumerate(sorted_values):
                    if value in self.models[column].index:
                        idx = self.models[column].index == value
                        flat[n * self.bin_count:(n + 1) * self.bin_count] = self.models[column][idx].to_numpy()[0]
            else:
                flat = self.models[column].to_numpy().reshape(-1)
            vectors.append(flat)
        return np.concatenate(vectors)

    def model_vectors(self, column_ordering=None, init_value=0):
        """
            column_ordering maps a column name to an ordered set of column values
            the function returns a two dimensional vector [p(x|f_c), d(x)] and column indices where
            p is the probability of success given column c has feature value f
            d is the density of events, sum(d) == 1 within one column
        """
        p_vectors = []
        d_vectors = []
        c_indices = [0]
        columns = sorted(self.columns)
        for column in columns:
            if column_ordering is not None:
                sorted_values = column_ordering[column]
                p_vector = np.zeros(len(sorted_values) * self.bin_count)
                p_vector[:] = init_value
                d_vector = np.zeros(len(sorted_values) * self.bin_count)
                d_vector[:] = init_value
                for n, value in enumerate(sorted_values):
                    if value in self.models[column].index:
                        idx = self.models[column].index == value
                        p_vector[n * self.bin_count:(n + 1) * self.bin_count] = \
                            self.models[column][idx].to_numpy()[:, 0]
                        d_vector[n * self.bin_count:(n + 1) * self.bin_count] = \
                            self.models[column][idx].to_numpy()[:, 1]
            else:
                p_vector = self.models[column].to_numpy()[:, 0]
                d_vector = self.models[column].to_numpy()[:, 1]
            c_indices.append(c_indices[-1] + d_vector.shape[0])
            p_vectors.append(p_vector)
            d_vectors.append(d_vector)
        return np.stack([np.concatenate(p_vectors), np.concatenate(d_vectors)]), np.array(c_indices)

    def transform(self, X, concatenate=False, mode='p'):
        result = []
        for column in X.columns:
            feature_column = X[[column]]
            column_model = feature_column.join(self.models[column], how='left', on=column)
            dat = column_model.reset_index()[column_model.columns[1:]].fillna(0).to_numpy()
            if 'p' in mode and 'd' in mode:
                pass
            elif 'p' in mode:
                dat = dat[:, 0]
            elif 'd' in mode:
                dat = dat[:, 1]
            result.append(dat)

        if concatenate:
            return np.concatenate(result, axis=1)
        return result

    def _to_bins(self, bins, model_attempts):
        bins = pd.qcut(model_attempts, bins, duplicates='drop', retbins=False)
        return bins.cat.codes

    def fit_transform(self, X, y, bin_count=1, concatenate=False, normalize=True):
        return self.fit(X, y, bin_count, normalize).transform(X, concatenate)


class Hist:
    def __init__(self):
        pass

    def p_hist_nd(self, X, y, bins=10, fill_na="mean"):
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        edges = np.linspace(start=0, stop=1, num=bins, endpoint=True)
        p, _ = np.histogramdd(X, bins=[edges for _ in range(X.shape[1])], density=True)
        # suc, _ = np.histogramdd(X[y == 1], bins=[edges for _ in range(X.shape[1])])
        # p = suc / cnt

        return p

    def to_p_hist(self, X, y, dim=1, bins=10):
        """
            X = nxm matrix, y=m vector
        """
        dims = np.arange(start=0, stop=X.shape[1])

        if dim == 1:
            for i in dims[:-1]:
                yield self.p_hist_nd(X[:, i], y, bins), (i)
        if dim == 2:
            for i in dims[:-1]:
                for j in dims[i + 1:]:
                    yield self.p_hist_nd(X[:, (i, j)], y, bins), (i, j)


if __name__ == '__main__':
    df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
                               'two'],
                       'bar': [1, 0, 0, 1, 1, 0],
                       'baz': [1, 2, 1, 2, 1, 3],
                       'zoo': ['x', 'x', 'x', 'z', 'q', 'q']})

    encoder = AggregateEncoder()
    encoder.fit(df.loc[:, [c for c in df.columns if c != 'bar']], df.loc[:, 'bar'], normalize=False)
    for name, model in encoder.models.items():
        print('------------')
        print(name)
        print('------------')
        print(model)
    encoder.transform(df[['foo', 'baz', 'zoo']])
    print(encoder.model_vectors(
        column_ordering={
            'foo': ['one', 'two'],
            'baz': [1, 2, 3],
            'zoo': ['p', 'x', 'z', 'q', 't']
        }
    ))
    print(encoder.model_vectors())

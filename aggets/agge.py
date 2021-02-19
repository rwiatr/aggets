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
        # print('-----------')
        # print(X)
        # print('-----------')
        for column in X.columns:
            feature_column = X[[column]]
            feature_column['y'] = y
            model = feature_column.groupby(column).agg(['sum', 'count'])
            model['p'] = model[('y', 'sum')] / model[('y', 'count')]
            # print('-----------')
            # print(feature_column)
            # print(column)
            # print(model[('y', 'count')])
            # print('-----------')
            model['bin'] = self._to_bins(bin_count, model[('y', 'count')])

            piv = model.pivot(columns='bin', values=('p', ''))
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
                flat = np.zeros(len(sorted_values))
                flat[:] = init_value
                for n, value in enumerate(sorted_values):
                    if value in self.models[column].index:
                        idx = self.models[column].index == value
                        flat[n] = self.models[column][idx].to_numpy()[0]
            else:
                flat = self.models[column].to_numpy().reshape(-1)
            vectors.append(flat)
        return np.concatenate(vectors)

    def transform(self, X, concatenate=False):
        result = []
        for column in X.columns:
            feature_column = X[[column]]
            column_model = feature_column.join(self.models[column], how='left', on=column)
            result.append(column_model.reset_index()[column_model.columns[1:]].fillna(0).to_numpy())

        if concatenate:
            return np.concatenate(result, axis=1)
        return result

    def _to_bins(self, bins, model_attempts):
        bins = pd.qcut(model_attempts, bins, duplicates='drop', retbins=False)
        return bins.cat.codes

    def fit_transform(self, X, y, bin_count=1, concatenate=False, normalize=True):
        return self.fit(X, y, bin_count, normalize).transform(X, concatenate)


if __name__ == '__main__':
    df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
                               'two'],
                       'bar': [1, 0, 0, 1, 1, 0],
                       'baz': [1, 2, 1, 2, 1, 3],
                       'zoo': ['x', 'x', 'x', 'z', 'q', 'q']})
    encoder = AggregateEncoder()
    encoder.fit(df.loc[:, [c for c in df.columns if c != 'bar']], df.loc[:, 'bar'])
    for name, model in encoder.models.items():
        print('------------')
        print(name)
        print('------------')
        print(model)

    print(encoder.model_vector(
        column_ordering={
            'foo': ['one', 'two'],
            'baz': [1, 2, 3],
            'zoo': ['p', 'x', 'z', 'q', 't']
        }
    ))
    print(encoder.model_vector())
#
# class LogisticAggregateEncoder:
#
#     def __init__(self):
#         self.models = {}
#
#     def fit(self, X, y, bin_count=1, normalize=True):
#         for column in X.columns:
#             feature_column = X[[column]]
#             feature_column['y'] = y
#             model = feature_column.groupby(column).agg(['sum', 'count'])
#             model['p'] = model[('y', 'sum')] / model[('y', 'count')]
#             model['bin'] = self._to_bins(bin_count, model[('y', 'count')])
#             piv = model.pivot(columns='bin', values=('p', ''))
#             zero_out = piv.isna()
#             if normalize:
#                 piv = ((piv - piv.min()) / (piv.max() - piv.min())).fillna(1)
#             piv[zero_out] = 0
#             model[piv.columns] = piv
#             self.models[column] = model[piv.columns]
#         return self
#
#     def transform(self, X, concatenate=False):
#         result = []
#         for column in X.columns:
#             feature_column = X[[column]]
#             column_model = feature_column.join(self.models[column], how='left', on=column)
#             result.append(column_model.reset_index()[column_model.columns[1:]].fillna(0).to_numpy())
#
#         if concatenate:
#             return np.concatenate(result, axis=1)
#         return result
#
#     def _to_bins(self, bins, model_attempts):
#         bins = pd.qcut(model_attempts, bins, duplicates='drop', retbins=False)
#         return bins.cat.codes
#
#     def fit_transform(self, X, y, bin_count=1, concatenate=False, normalize=True):
#         return self.fit(X, y, bin_count, normalize).transform(X, concatenate)

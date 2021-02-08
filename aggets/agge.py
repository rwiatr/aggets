import numpy as np
import pandas as pd

"""
Source: https://github.com/rwiatr/agge/
"""


class AggregateEncoder:

    def __init__(self):
        self.models = {}

    def fit(self, X, y, bin_count=1, normalize=True):
        for column in X.columns:
            feature_column = X[[column]]
            feature_column['y'] = y
            model = feature_column.groupby(column).agg(['sum', 'count'])
            model['p'] = model[('y', 'sum')] / model[('y', 'count')]
            model['bin'] = self._to_bins(bin_count, model[('y', 'count')])
            piv = model.pivot(columns='bin', values=('p', ''))
            zero_out = piv.isna()
            if normalize:
                piv = ((piv - piv.min()) / (piv.max() - piv.min())).fillna(1)
            piv[zero_out] = 0
            model[piv.columns] = piv
            self.models[column] = model[piv.columns]
        return self

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

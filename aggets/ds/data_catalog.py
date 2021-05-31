import os
import wget
import pandas as pd

from aggets import util


class Binary:
    def __init__(self, path='/tmp/ds'):
        self.path = path

    def _load(self, file, set_name, zipped=False):
        source = f'https://github.com/scikit-multiflow/streaming-datasets/raw/master/{set_name}'
        if not os.path.isfile(file):
            wget.download(url=source, out=file)
        df = pd.read_csv(file)
        print(file, df.shape, df.columns)
        n = len(df)
        train_df = df[0:int(n * 0.7)]
        val_df = df[int(n * 0.7):int(n * 0.9)]
        test_df = df[int(n * 0.9):]

        num_features = df.shape[1]
        column_indices = {name: i for i, name in enumerate(df.columns)}

        return {'train': train_df, 'val': val_df, 'df': df,
                'test': test_df, 'features': num_features,
                'column_indices': column_indices}

    def agr_a(self):
        set_name = 'agr_a.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def agr_g(self):
        set_name = 'agr_g.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def hyper_f(self):
        set_name = 'hyper_f.csv.zip'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name, zipped=True)

    def airlines(self):
        set_name = 'airlines.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def electric(self):
        set_name = 'elec.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def iris_ts(self):
        set_name = 'iris_timestamp.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def sea_a(self):
        set_name = 'sea_a.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def sea_g(self):
        set_name = 'sea_g.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def weather(self):
        set_name = 'weather.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)



class Binary_pp:
    def __init__(self, path='/tmp/ds'):
        self.path = path

    def _load(self, file, set_name, zipped=False):
        source = f'https://github.com/rlyyah/concept_drift_ds/tree/master/data/{set_name}'
        if not os.path.isfile(file):
            wget.download(url=source, out=file)
        df = pd.read_csv(file)
        print(file, df.shape, df.columns)
        n = len(df)
        train_df = df[0:int(n * 0.7)]
        val_df = df[int(n * 0.7):int(n * 0.9)]
        test_df = df[int(n * 0.9):]

        num_features = df.shape[1]
        column_indices = {name: i for i, name in enumerate(df.columns)}

        return {'train': train_df, 'val': val_df, 'df': df,
                'test': test_df, 'features': num_features,
                'column_indices': column_indices}

    def covtype(self):
        set_name = 'covtype.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def fin_adult(self):
        set_name = 'first_fin_adult.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def fin_bank(self):
        set_name = 'first_fin_bank.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def fin_digits(self):
        set_name = 'first_fin_digits08.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def fin_digits_bis(self):
        set_name = 'first_fin_digits17.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def fin_musk(self):
        set_name = 'first_fin_musk.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def fin_phis(self):
        set_name = 'first_fin_phis.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)
    
    def fin_wine(self):
        set_name = 'first_fin_wine.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def nsl(self):
        set_name = 'nsl.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def nsl_kdd(self):
        set_name = 'nsl_kdd.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def phishing(self):
        set_name = 'phishing.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def spam(self):
        set_name = 'spam.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

    def spamassassian(self):
        set_name = 'spamassassian.csv'
        file = f'{self.path}/{set_name}'
        return self._load(file, set_name)

if __name__ == '__main__':
    binary = Binary()
    # binary.agr_a()
    # binary.agr_g()
    # binary.sea_a()
    # binary.sea_g()
    # binary.weather()
    # binary.electric()

    import aggets.ds.aggregate_nd as agg_nd


    def make_window(data, name, window_size=50):
        train_np = data['train']
        val_np = data['val']
        test_np = data['test']
        file_name = f'{name}-ws{window_size}.bin'
        if not os.path.exists(file_name):
            window = agg_nd.window_generator(train_np.to_numpy(), val_np.to_numpy(), test_np.to_numpy(),
                                             window_size=window_size, e=0.00001, hist_bins=20, hist_dim=1)
            window.init_structures()
            util.save(window, path=file_name)
        return util.load(path=file_name)


    data_types = {
        'agr_a': make_window(binary.agr_a(), 'agr_a', window_size=500),
        'agr_g': make_window(binary.agr_g(), 'agr_g', window_size=500),
        'sea_a': make_window(binary.sea_a(), 'sea_a', window_size=500),
        'sea_g': make_window(binary.sea_g(), 'sea_g', window_size=500),
        'weather': make_window(binary.weather(), 'weather'),
        'electric': make_window(binary.electric(), 'electric')
    }

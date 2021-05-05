import pandas as pd
from sklearn import preprocessing


def load(path, normalize=0, train_split=0.7, val_split=0.9):
    df = pd.read_csv(path).astype('float32')

    if normalize == 1:
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, columns=df.columns)

    n = len(df)
    train_df = df[0:int(n * train_split)]
    val_df = df[int(n * train_split):int(n * val_split)]
    test_df = df[int(n * val_split):]

    num_features = df.shape[1]
    column_indices = {name: i for i, name in enumerate(df.columns)}

    return {'train': train_df, 'val': val_df, 'df': df,
            'test': test_df, 'features': num_features,
            'column_indices': column_indices}

import pandas as pd


def load(path='data/ts/hyperplane'):
    """
    class is the label [0,1]
    """
    df = pd.read_csv(f'{path}/hyper_f.csv').astype('float32')

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    num_features = df.shape[1]
    column_indices = {name: i for i, name in enumerate(df.columns)}

    return {'train': train_df, 'val': val_df, 'df': df,
            'test': test_df, 'features': num_features,
            'column_indices': column_indices}


def load_discrete(bins=10, path='data/ts/hyperplane'):
    """
    class is the label [0,1]
    """
    df = pd.read_csv(f'{path}/hyper_f.csv').astype('float32')
    atts = [column for column in df.columns if column != 'class']
    discrete = df.copy()
    for att in atts:
        discrete[att] = (discrete[att] * bins).astype(int) / bins

    n = len(discrete)
    train_ddf = discrete[0:int(n * 0.7)]
    val_ddf = discrete[int(n * 0.7):int(n * 0.9)]
    test_ddf = discrete[int(n * 0.9):]

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    num_features = df.shape[1]
    column_indices = {name: i for i, name in enumerate(df.columns)}

    return {
        'df': {'train': train_df,
               'val': val_df,
               'test': test_df},
        'ddf': {'train': train_ddf,
                'val': val_ddf,
                'test': test_ddf},
        'features': num_features,
        'column_indices': column_indices}


def load_df():
    """
    class is the label [0,1]
    """
    df = pd.read_csv('data/ts/hyperplane/hyper_f.csv').astype('float32')
    # df['class0'] = 1 - df['class']
    # df['class1'] = df['class']
    # df = df.drop(columns=['class'])

    column_indices = {name: i for i, name in enumerate(df.columns)}

    return {'df': df,
            'column_indices': column_indices}


def load_discrete_df(bins=10):
    """
    class is the label [0,1]
    """
    df = pd.read_csv('data/ts/hyperplane/hyper_f.csv').astype('float32')

    atts = [column for column in df.columns if column != 'class']
    for att in atts:
        df[att] = (df[att] * bins).astype(int) / bins

    column_indices = {name: i for i, name in enumerate(df.columns)}

    return {'df': df,
            'column_indices': column_indices}

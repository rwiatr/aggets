import pandas as pd


def load():
    """
    class is the label [0,1,2,3]
    """
    df = pd.read_csv('data/ts/hyperplane/hyper_f.csv')

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    num_features = df.shape[1]
    column_indices = {name: i for i, name in enumerate(df.columns)}

    return {'train': train_df, 'val': val_df,
            'test': test_df, 'features': num_features,
            'column_indices': column_indices}

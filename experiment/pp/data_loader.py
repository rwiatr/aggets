import pandas as pd

def create_data_object_csv(path, train_p, vali_p, test_p):
    """
    csv files loader 
    """
    data_test = pd.read_csv(path, index_col=0)
    n = data_test.shape[0]
    num_features = data_test.shape[1]

    train_df, vali_df, test_df = data_test[:int(n*train_p)], data_test[int(n*train_p):int(n*(train_p+vali_p))], data_test[int(n*(train_p+vali_p)):]
    column_indices = {name: i for i, name in enumerate(data_test.columns[:-1])}
    cols = ['att' + str(i + 1) for i, name in enumerate(data_test.columns[:-1])]

    data_obj = {
        'train': train_df, 
        'val': vali_df, 
        'test': test_df, 
        'features': num_features, 
        'column_indices': column_indices , 
        'df': data_test, 
        'cols': cols}

    return data_obj

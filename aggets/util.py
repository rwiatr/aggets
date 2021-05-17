import pickle

def save(wg, path='file.bin'):
    with open(path, 'wb') as handle:
        pickle.dump(wg, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(path='file.bin'):
    with open(path, 'rb') as handle:
        wg = pickle.load(handle)
    return wg


def data_to_device(data, device):
    if isinstance(data, list):
        data = [x.to(device) for x in data]
    else:
        data = data.to(device)
    return data

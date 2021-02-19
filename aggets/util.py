import pickle

def save(wg, path='file.bin'):
    with open(path, 'wb') as handle:
        pickle.dump(wg, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(path='file.bin'):
    with open(path, 'rb') as handle:
        wg = pickle.load(handle)
    return wg

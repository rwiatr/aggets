import pickle
import torch


def save(wg, path='file.bin'):
    with open(path, 'wb') as handle:
        pickle.dump(wg, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(path='file.bin'):
    with open(path, 'rb') as handle:
        wg = pickle.load(handle)
    return wg


def cuda_if_possible():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def float_tensor_default():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

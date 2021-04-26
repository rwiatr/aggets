from aggets.ds.lerners import LogisticLerner
import numpy as np


def test_correct_dimensionality():
    lerner = LogisticLerner(samples=5, sample_frac=0.1)
    windows_np = [np.random.rand(100, 4), np.random.rand(100, 4), np.random.rand(100, 4)]
    models = lerner.init_models(windows_np)

    """ 3 windows, 5 samples, 3 X dimensions + bias """
    assert models.shape == (3, 5, 4)


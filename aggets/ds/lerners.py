from sklearn import linear_model
import numpy as np
import torch


class LogisticLerner:

    def __init__(self, samples, sample_frac):
        self.samples = samples
        self.sample_frac = sample_frac

    def init_models(self, windows_np):
        """
            windows_np is an array of training examples
        """
        print('calculating models ...', end='\r')
        lrs = []

        for n, window in enumerate(windows_np):
            lr = []
            for attempt in range(self.samples):
                sampling = np.random.randint(window.shape[0], size=int(window.shape[0] * self.sample_frac + 1))
                sample = window[sampling]
                X = sample[:, :-1]
                y = sample[:, -1]

                fit = linear_model.LinearRegression().fit(X, y)
                lr_vec = np.zeros(fit.coef_.shape[0] + 1)
                lr_vec[:-1] = fit.coef_.reshape(-1)
                lr_vec[-1] = fit.intercept_
                lr.append(torch.Tensor(lr_vec))
            print(f'calculating models ... {n}/{len(windows_np)}', end='\r')

            lrs.append(torch.stack(lr))

        print(f'                                                         ', end='\r')

        """ (window, attempt, lr_vec) """
        return torch.stack(lrs)

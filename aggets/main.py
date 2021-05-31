import os
import torch.nn as nn

import aggets.train as train
import aggets.ds.aggregate_nd as agg_nd
import aggets.util as util
import aggets.ds.hyper_f_load as hyper_f_load
import matplotlib.pyplot as plt

from aggets.model.aggregate import WindowConfig
import aggets.model.aggregate2 as agg_m
import aggets.model.simple as simple
import aggets.model.fourier as fourier

if __name__ == '__main__':
    def load_data():
        data = hyper_f_load.load(path='../experiment/data/ts/hyperplane')
        cols = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10']
        train_np = agg_nd.as_np(data['train'], cols, 'class')
        val_np = agg_nd.as_np(data['val'], cols, 'class')
        test_np = agg_nd.as_np(data['test'], cols, 'class')

        if not os.path.exists('window_data.bin'):
            window = agg_nd.window_generator(train_np, val_np, test_np, window_size=50,
                                             e=0.00001, hist_bins=20, hist_dim=1)
            window.init_structures()
            util.save(window, path='window_data.bin')

        return util.load(path='window_data.bin')  # 20 bins, 1 dim, 500ws


    def lstm_all_to_lr(w, num_layers, hidden):
        # HIST+LR -> LR
        inp = nn.Sequential(
            agg_m.FlatCat(),
            simple.mlp(features=411, num_layers=1, out_features=hidden)
        )
        out = simple.mlp(features=hidden, num_layers=2, out_features=11)

        lstm = agg_m.AutoregLstm(input=inp, output=out, in_len=10, out_len=5, hidden=hidden, num_layers=num_layers)
        # -------
        lstm.name = 'autoreg-lstm'
        train.train_window_models([lstm], w, patience=10, validate=True, weight_decay=0, max_epochs=1,
                                  lrs=[0.0001, 0.00001],
                                  source='all', target='lr', log=False)
        # -------
        _, axs = plt.subplots(ncols=3, nrows=1, sharey='row', figsize=(15, 3))
        w.plot_lr(axs=axs)
        w.plot_model(lstm, axs=axs, other={'source': 'all', 'target': 'lr'})
        plt.show()


    def fourier_hist0_to_hist0(w):
        frr = fourier.HistogramLerner(extra_dims=1, t_in=10)
        frr = fourier.FAdapter3(frr)
        frr.window_config = WindowConfig(output_sequence_length=1, input_sequence_length=10, label_stride=1)
        # -------
        frr.name = 'fourier'
        train.train_window_models([frr], w, patience=2, validate=True, weight_decay=0, max_epochs=1,
                                  lrs=[0.001, 0.0001],
                                  source='agg[0]', target='agg[0]', log=False)

        # ------- p hist
        _, axs = plt.subplots(ncols=3, nrows=1, sharey='row', figsize=(15, 3))
        w.plot_agg_dist(axs=axs, select=lambda a: a[:, 0, 0])
        w.plot_model_agg_dist(model=frr, axs=axs, other={'source': 'agg[0]', 'target': 'agg[0]'})
        plt.show()


    w = load_data()
    lstm_all_to_lr(w, num_layers=1, hidden=256)
    fourier_hist0_to_hist0(w)

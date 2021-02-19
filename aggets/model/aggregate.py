import torch
import torch.nn as nn

from aggets.model import simple


class LrConv(nn.Module):
    def __init__(self, conv_features, conv_width, lr_features, conv_hidden=32, mlp_layers=3, resid=True):
        super(LrConv, self).__init__()
        self.conv_width = conv_width
        self.resid = resid
        self.conv = nn.Sequential(simple.conv_1d(conv_width=conv_width,
                                                 features=conv_features,
                                                 hidden=conv_hidden,
                                                 out_features=conv_hidden),
                                  nn.ReLU())
        self.mlp = simple.mlp(features=lr_features + conv_hidden,
                              num_layers=mlp_layers,
                              hidden=lr_features + conv_hidden,
                              out_features=lr_features)

    def forward(self, x):
        x, lr = x
        lr = lr[:, self.conv_width - 1:, :]
        x = self.conv(x)
        cat = torch.cat([x, lr], dim=2)
        x = self.mlp(cat)

        if self.resid:
            return lr.detach() + x
        else:
            return x


"""
CREATE CONTINUOUS QUERY "transactions"
ON "orders"
BEGIN
  SELECT mean(value) as value,
  INTO "downsampled.cpu_load"
  FROM cpu_load
  GROUP BY time(1h), type, order_id
END
"""


class LrNConv(nn.Module):
    def __init__(self, ts_features, lr_features,
                 conv_layers=3,
                 conv_width=8,
                 ts_conv_pool_width=3,
                 ts_conv_pool_stride=3,
                 lr_conv_pool_width=3,
                 lr_conv_pool_stride=3,
                 ts_conv_fc_layers=1,
                 lr_conv_fc_layers=1,
                 ts_conv_out_feature_div=2,
                 lr_conv_out_feature_div=2,
                 mlp_width=10,
                 mlp_layers=3,
                 out_features=1):
        super(LrNConv, self).__init__()
        self.conv_width = conv_width
        self.conv_ts = simple.n_conv_1d(features=ts_features,
                                        conv_layers=conv_layers,
                                        pool_width=ts_conv_pool_width,
                                        pool_stride=ts_conv_pool_stride,
                                        fc_layers=ts_conv_fc_layers,
                                        conv_width=conv_width,
                                        conv_out_feature_div=ts_conv_out_feature_div,
                                        out_features=mlp_width)
        self.conv_lr = simple.n_conv_1d(features=lr_features,
                                        conv_layers=conv_layers,
                                        pool_width=lr_conv_pool_width,
                                        pool_stride=lr_conv_pool_stride,
                                        fc_layers=lr_conv_fc_layers,
                                        conv_width=conv_width,
                                        conv_out_feature_div=lr_conv_out_feature_div,
                                        out_features=mlp_width)
        self.mlp = simple.mlp(features=mlp_width * 2,
                              num_layers=mlp_layers,
                              hidden=mlp_width * 2,
                              out_features=out_features)

    def forward(self, x):
        ts, lr = x
        original_lr = lr
        # print(lr.shape, ts.shape)
        ts = self.conv_ts(ts)
        lr = self.conv_lr(lr)
        # print(lr.shape, ts.shape)
        cat = torch.cat([ts, lr], dim=2)
        # print(cat.shape)
        x = self.mlp(cat)

        # data is [batch, sequence, lr]
        original_lr = original_lr[:, -x.shape[1]:, :].detach()  # take last x.shape[1]
        # print(original_lr.shape, x.shape)
        return original_lr + x


class CombinedLrNConv(nn.Module):
    def __init__(self, ts_features, lr_features,
                 conv_layers=3,
                 conv_width=8,
                 conv_pool_width=3,
                 conv_pool_stride=3,
                 conv_fc_layers=1,
                 conv_out_feature_div=2,
                 out_features=1):
        super(CombinedLrNConv, self).__init__()
        self.conv_width = conv_width
        self.conv = simple.n_conv_1d(features=ts_features + lr_features,
                                     conv_layers=conv_layers,
                                     pool_width=conv_pool_width,
                                     pool_stride=conv_pool_stride,
                                     fc_layers=conv_fc_layers,
                                     conv_width=conv_width,
                                     out_features=out_features,
                                     conv_out_feature_div=conv_out_feature_div)

    def forward(self, x):
        ts, lr = x
        x = self.conv(torch.cat([ts, lr], dim=2))
        lr = lr[:, -x.shape[1]:, :].detach()  # take last x.shape[1]
        return lr + x

import math

import torch
import torch.nn as nn

from aggets.model import simple


class WindowConfig:
    def __init__(self, output_sequence_length=None, input_sequence_length=None, label_stride=1):
        self.output_sequence_length = output_sequence_length
        self.input_sequence_length = input_sequence_length
        self.label_stride = label_stride

    def with_in_seq(self, input_sequence_length):
        return WindowConfig(output_sequence_length=self.output_sequence_length,
                            input_sequence_length=input_sequence_length,
                            label_stride=self.label_stride)

    def with_stride(self, label_stride):
        return WindowConfig(output_sequence_length=self.output_sequence_length,
                            input_sequence_length=self.input_sequence_length,
                            label_stride=label_stride)


class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.mlp = simple.mlp(1)
        self.window_config = WindowConfig()

    def forward(self, x):
        x, lr = x
        return lr[:, -1:]


class DummyNetTS(nn.Module):
    def __init__(self, dist):
        super(DummyNetTS, self).__init__()
        self.mlp = simple.mlp(1)
        self.window_config = WindowConfig()
        self.dist = dist

    def forward(self, x):
        x, lr = x
        return x[:, self.dist]


class LrConvOld(nn.Module):
    def __init__(self, conv_features, conv_width, lr_features, conv_hidden=32, mlp_layers=3):
        super(LrConvOld, self).__init__()
        self.conv_width = conv_width
        self.window_config = WindowConfig(input_sequence_length=conv_width, output_sequence_length=1)
        self.conv = simple.conv_1d(conv_width=conv_width,
                                   features=conv_features,
                                   hidden=conv_hidden,
                                   out_features=conv_hidden)
        self.mlp = simple.mlp(features=lr_features + conv_hidden,
                              num_layers=mlp_layers,
                              hidden=lr_features + conv_hidden,
                              out_features=lr_features)

    def forward(self, x):
        x, lr = x
        # batch, seq, features
        lr = lr[:, self.conv_width - 1:, :]
        x = self.conv(x)
        cat = torch.cat([x, lr], dim=2)
        # x = [32, 1, 204]
        # lr = [32, 1, 11]
        # cat = [32, 1, 215]
        x = self.mlp(cat)  # delta between last regression and next one
        return lr + x


class LrConv(nn.Module):
    def __init__(self,
                 conv_features,
                 conv_width,
                 lr_features,
                 conv_out=32,
                 conv_layers=1,
                 conv_arg_map={},
                 mlp_layers=3,
                 mlp_arg_map={},
                 resid=True):
        super(LrConv, self).__init__()
        self.conv_width = conv_width
        self.window_config = WindowConfig(input_sequence_length=conv_width, output_sequence_length=1)
        self.resid = resid
        self.conv, conv_seq_reduction = simple.n_conv_1d(features=conv_features,
                                                         conv_width=conv_width,
                                                         out_features=conv_out,
                                                         conv_layers=conv_layers,
                                                         return_seq_reducer=True,
                                                         **conv_arg_map)
        self.min_input_dim = conv_seq_reduction.minimal_input_dim
        self.mlp = simple.mlp(features=lr_features + conv_out,
                              num_layers=mlp_layers,
                              hidden=lr_features + conv_out,
                              out_features=lr_features,
                              **mlp_arg_map)

    def forward(self, x):
        x, lr = x
        x = self.conv(x)
        lr = lr[:, -x.shape[1]:, :]
        cat = torch.cat([x, lr], dim=2)
        x = self.mlp(cat)

        if self.resid:
            return lr.detach() + x
        else:
            return x


class LstmLr(nn.Module):
    def __init__(self, ts_features, lr_features, num_layers=1, hidden=64):
        super(LstmLr, self).__init__()
        self.window_config = WindowConfig(output_sequence_length=1)

        hidden_shape = (ts_features + lr_features) if hidden is None else hidden
        self.lstm = simple.lstm(features=ts_features + lr_features,
                                hidden=hidden_shape,
                                out_features=lr_features,
                                num_layers=num_layers)

    def forward(self, x):
        ts, lr = x
        x = torch.cat([ts, lr], dim=2)
        x = self.lstm(x) + lr[:, -1:, :]
        return x


class AutoregLstmLr(nn.Module):
    def __init__(self, ts_features, lr_features, num_layers=1, hidden=64, output_sequence_length=1,
                 return_deltas=False):
        super(AutoregLstmLr, self).__init__()
        self.window_config = WindowConfig(output_sequence_length=output_sequence_length)
        self.return_deltas = return_deltas
        hidden_shape = (ts_features + lr_features) if hidden is None else hidden
        self.enc = nn.LSTM(input_size=ts_features + lr_features,
                           hidden_size=hidden_shape,
                           num_layers=num_layers, batch_first=True)
        self.dec = nn.LSTM(input_size=hidden_shape,
                           hidden_size=hidden_shape,
                           num_layers=num_layers, batch_first=True)
        self.to_lr = simple.mlp(features=hidden_shape, out_features=lr_features)

    def decode(self, enc, lr):
        for _ in range(self.window_config.output_sequence_length):
            enc, _ = self.dec(enc)
            delta = self.to_lr(enc)
            lr = delta + lr
            yield lr, delta

    def forward(self, x):
        _, lr = x
        x = torch.cat(x, dim=2)
        x = self.encode(x)
        x = self.decode(x, lr[:, -1:, :])

        lrs = []
        deltas = []
        for lr, delta in x:
            lrs.append(lr)
            deltas.append(delta)

        if not self.return_deltas:
            return torch.cat(lrs, dim=1)
        return torch.cat(lrs, dim=1), torch.cat(deltas, dim=1)

    def encode(self, x):
        enc, _ = self.enc(x)
        return enc[:, -1:, :]


class AutoregLstmLrAux(nn.Module):
    def __init__(self, ts_features, lr_features, num_layers=1, hidden=64, output_sequence_length=1,
                 return_deltas=False, single_output=False, ret_type='all'):
        super(AutoregLstmLrAux, self).__init__()
        self.window_config = WindowConfig(output_sequence_length=output_sequence_length)
        self.return_deltas = return_deltas
        hidden_shape = (ts_features + lr_features) if hidden is None else hidden
        self.enc = nn.LSTM(input_size=ts_features + lr_features,
                           hidden_size=hidden_shape,
                           num_layers=num_layers, batch_first=True)
        self.dec = nn.LSTM(input_size=hidden_shape,
                           hidden_size=hidden_shape,
                           num_layers=num_layers, batch_first=True)
        self.single_output = single_output
        if single_output:
            self.to_delta_aux = simple.mlp(features=hidden_shape, out_features=lr_features + ts_features)
        else:
            self.to_aux = simple.mlp(features=hidden_shape, out_features=ts_features)
            self.to_delta = simple.mlp(features=hidden_shape, out_features=lr_features)
        self.lr_features = lr_features
        self.ret_type = ret_type

    def decode(self, enc, lr):
        for _ in range(self.window_config.output_sequence_length):
            enc, _ = self.dec(enc)
            if self.single_output:
                delta_aux = self.to_delta_aux(enc)
                delta = delta_aux[:, :, :self.lr_features]
                aux = delta_aux[:, :, self.lr_features:]
            else:
                aux = self.to_aux(enc)
                delta = self.to_delta(enc)
            lr = delta + lr
            if self.ret_type == 'lr':
                yield lr, delta
            elif self.ret_type == 'aux':
                yield aux, delta
            else:
                yield torch.cat([aux, lr], dim=-1), delta

    def forward(self, x):
        _, lr = x
        x = torch.cat(x, dim=2)
        x = self.encode(x)
        x = self.decode(x, lr[:, -1:, :])

        lrs = []
        deltas = []
        for lr, delta in x:
            lrs.append(lr)
            deltas.append(delta)

        if not self.return_deltas:
            return torch.cat(lrs, dim=1)
        return torch.cat(lrs, dim=1), torch.cat(deltas, dim=1)

    def encode(self, x):
        enc, _ = self.enc(x)
        return enc[:, -1:, ]


class AutoregLstm(nn.Module):
    def __init__(self, ts_features, num_layers=1, hidden=64, output_sequence_length=1):
        super(AutoregLstm, self).__init__()
        self.window_config = WindowConfig(output_sequence_length=output_sequence_length)
        hidden_shape = ts_features if hidden is None else hidden
        self.enc = nn.LSTM(input_size=ts_features,
                           hidden_size=hidden_shape,
                           num_layers=num_layers, batch_first=True)
        self.dec = nn.LSTM(input_size=hidden_shape,
                           hidden_size=hidden_shape,
                           num_layers=num_layers, batch_first=True)
        self.to_result = simple.mlp(features=hidden_shape, out_features=ts_features)

    def decode(self, enc):
        for _ in range(self.window_config.output_sequence_length):
            enc, _ = self.dec(enc)
            yield self.to_result(enc)

    def forward(self, x):
        x, _ = x
        x = self.encode(x)
        x = self.decode(x)

        return torch.cat(list(x), dim=1)

    def encode(self, x):
        enc, _ = self.enc(x)
        return enc[:, -1:, ]


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
        self.window_config = WindowConfig(input_sequence_length=conv_width, output_sequence_length=1)

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
                 out_features=1,
                 arg_map=None):
        super(CombinedLrNConv, self).__init__()
        arg_map = {} if arg_map is None else arg_map
        self.conv_width = conv_width
        self.window_config = WindowConfig(input_sequence_length=conv_width, output_sequence_length=1)
        self.conv = simple.n_conv_1d(features=ts_features + lr_features,
                                     conv_layers=conv_layers,
                                     pool_width=conv_pool_width,
                                     pool_stride=conv_pool_stride,
                                     fc_layers=conv_fc_layers,
                                     conv_width=conv_width,
                                     out_features=out_features,
                                     conv_out_feature_div=conv_out_feature_div,
                                     **arg_map)

    def forward(self, x):
        ts, lr = x
        x = self.conv(torch.cat([ts, lr], dim=2))
        lr = lr[:, -x.shape[1]:, :].detach()  # take last x.shape[1]
        return lr + x


class CombinedLrNConvWithRawLr(nn.Module):
    def __init__(self, ts_features, lr_features,
                 conv_layers=3,
                 conv_width=8,
                 conv_pool_width=3,
                 conv_pool_stride=3,
                 conv_out_feature_div=2,
                 conv_out_features=1,
                 conv_arg_map=None,
                 mlp_layers=1,
                 mlp_arg_map=None):
        super(CombinedLrNConvWithRawLr, self).__init__()
        conv_arg_map = {} if conv_arg_map is None else conv_arg_map
        mlp_arg_map = {} if mlp_arg_map is None else mlp_arg_map

        self.conv_width = conv_width
        self.window_config = WindowConfig(input_sequence_length=conv_width, output_sequence_length=1)

        self.conv = simple.n_conv_1d(features=ts_features + lr_features,
                                     conv_layers=conv_layers,
                                     pool_width=conv_pool_width,
                                     pool_stride=conv_pool_stride,
                                     fc_layers=1,
                                     conv_width=conv_width,
                                     out_features=conv_out_features,
                                     conv_out_feature_div=conv_out_feature_div,
                                     **conv_arg_map)
        self.mlp = simple.mlp(features=conv_out_features + lr_features,
                              hidden=conv_out_features + lr_features,
                              num_layers=mlp_layers,
                              out_features=lr_features,
                              **mlp_arg_map)

    def forward(self, x):
        ts, lr = x
        x = self.conv(torch.cat([ts, lr], dim=2))
        # print(x.shape, lr[:, -x.shape[1]:, :].detach().shape)
        x = self.mlp(torch.cat([x, lr[:, -x.shape[1]:, :].detach()], dim=2))
        lr = lr[:, -x.shape[1]:, :].detach()  # take last x.shape[1]
        return lr + x


"""source: https://github.com/pytorch/examples/blob/master/word_language_model/model.py"""


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


"""source: https://github.com/pytorch/examples/blob/master/word_language_model/model.py"""


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ninp, nhead, nhid, nlayers, nout, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, nout)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        x, lr = src
        lr = torch.cat([lr[:, -1:, :]] * x.shape[1], dim=1)
        # print(x.shape, lr.shape)
        src = torch.cat([x, lr], dim=2)
        src = src[:, -56:, :]
        src = torch.transpose(src, 0, 1)
        padded = torch.zeros((src.shape[0], src.shape[1], self.ninp))
        padded[:, :, :src.shape[2]] = src
        src = padded

        if has_mask and False:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = src * math.sqrt(self.ninp)

        # print(src.shape)
        # print(torch.transpose(src, 0, 1).shape)

        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, self.src_mask)
        src = torch.transpose(src, 0, 1)  ### CZY JA MAM DOBRZE TE WYMIARY!!!!
        # print(src.shape)
        src = self.decoder(src)
        # print(src.shape)
        return src


class SimpleRNN(nn.Module):
    def __init__(self, lstm_props={}, mlp_props={}):
        lstm_props['batch_first'] = True
        self.rnn = nn.LSTM(**lstm_props)
        self.mlp = simple.mlp(**mlp_props)

    def forward(self, x):
        x, lr = x
        last = self.rnn(x)
        return self.mlp(torch.cat([last, lr], dim=1))

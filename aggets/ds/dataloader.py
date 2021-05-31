import torch.utils.data as data
import torch
import numpy as np

DEFAULT_BATCH = 32


class SimpleDataLoaderFactory:

    def __init__(self, debug=False):
        self.debug = debug

    def data_loader(self, aggregates, models, in_len, out_len, target='lr', source='all', agg='full',
                    y_offset=0, shuffle=True, batch_size=DEFAULT_BATCH, first_sample=False):
        """
            aggregates: [window, sample, ???]
            models: [window, sample, ???]
        """

        assert models.shape[0] == aggregates.shape[0] and models.shape[1] == aggregates.shape[1]
        windows, samples = aggregates.shape[0], aggregates.shape[1]
        if first_sample:
            samples = 1

        if agg == 'p':
            aggregates = aggregates[:, :, :0]

        if agg == 'd':
            aggregates = aggregates[:, :, 1:]

        aggregates = aggregates.transpose(0, 1)
        models = models.transpose(0, 1)

        class InSituDataSet(data.Dataset):
            def __getitem__(self, index):
                # iterate window wise
                # index = index % windows
                # sample = index // windows
                # iterate sample wise
                sample = index % samples
                index = index // samples

                start = index
                middle = start + in_len
                end = middle + out_len
                aggregates_in = aggregates[sample, start:middle]
                models_in = models[sample, start:middle]
                aggregates_out = aggregates[sample, middle + y_offset:end + y_offset]
                models_out = models[sample, middle + y_offset:end + y_offset]

                if target == 'lr':
                    output = models_out
                elif target == 'agg[0]':
                    output = aggregates_out[:, 0]
                elif target == 'agg':
                    output = aggregates_out
                elif target == 'all':
                    output = (aggregates_out, models_out)

                if source == 'lr':
                    input = models_in
                elif source == 'agg[0]':
                    input = aggregates_in[:, 0]
                elif source == 'agg':
                    input = aggregates_in
                elif source == 'all':
                    input = (aggregates_in, models_in)

                return input, output

            def __len__(self):
                return (windows - in_len - out_len - y_offset) * samples

        return data.DataLoader(InSituDataSet(), shuffle=shuffle, batch_size=batch_size)


class BasicDataLoaderCreator:

    def __init__(self, debug):
        self.debug = debug

    def compress_into_single_matrix(self, agg, lrs):
        """
            agg        = (window, attempt, aggregate_vec)
            lrs        = (window, attempt, lr)
            """
        agg = torch.transpose(agg, 0, 1)
        lrs = torch.transpose(lrs, 0, 1)
        """
            agg        = (attempt/batch, window/seq, aggregate_vec)
            lrs        = (attempt/batch, window/seq, lr)
            """
        if self.mark_type:  # TODO TEST IT
            ids_2 = torch.Tensor(np.linspace(0, agg.shape[2] - 1, agg.shape[2]))
            ids_2 = ids_2.repeat(1, agg.shape[0], agg.shape[1], 1).permute(1, 2, 3, 0)
            agg = torch.cat([ids_2, agg], dim=-1)

        if self.debug_output:
            ids_0 = torch.Tensor(np.linspace(0, agg.shape[0] - 1, agg.shape[0]))
            ids_1 = torch.Tensor(np.linspace(0, agg.shape[1] - 1, agg.shape[1]))
            ids_2 = torch.Tensor(np.linspace(0, agg.shape[2] - 1, agg.shape[2]))

            ids_0 = ids_0.repeat(1, agg.shape[1], agg.shape[2], 1).permute(3, 1, 2, 0)
            ids_1 = ids_1.repeat(1, agg.shape[0], agg.shape[2], 1).permute(1, 3, 2, 0)
            ids_2 = ids_2.repeat(1, agg.shape[0], agg.shape[1], 1).permute(1, 2, 3, 0)

            agg = torch.cat([ids_1, ids_0, ids_2, agg], dim=-1)

            ids_0 = torch.Tensor(np.linspace(0, lrs.shape[0] - 1, lrs.shape[0]))
            ids_1 = torch.Tensor(np.linspace(0, lrs.shape[1] - 1, lrs.shape[1]))

            ids_0 = ids_0.repeat(1, lrs.shape[1], 1).permute(2, 1, 0)
            ids_1 = ids_1.repeat(1, lrs.shape[0], 1).permute(1, 2, 0)
            lrs = torch.cat([ids_1, ids_0, lrs], dim=-1)
        agg = agg.reshape(agg.shape[0], agg.shape[1], -1)
        blob = torch.cat([agg, lrs], dim=-1)

        """
            blob = [attempt, window, data]
            data = [(ids)p, (ids)d, (ids)lr]
        """
        return blob, agg.shape[-1], lrs.shape[-1]

    def _split_window(self,
                      lr_size,
                      agg_size,
                      in_len, out_len,
                      return_density=False,
                      single_ts=True,
                      target_histograms=False,
                      target_regression=True):

        def __split_window(features):
            """ features = [attempt, time, (p, d, lr), type]"""
            p_size = agg_size // 2
            source = features[:, :in_len, :, 0]
            target = features[:, -out_len:, :, 1]
            aggregates_p = source[:, :, :p_size]
            aggregates_d = source[:, :, p_size:agg_size]
            source = source[:, :, agg_size:agg_size + lr_size]

            if target_histograms and not target_regression:
                target = target[:, :, :p_size]
            elif target_histograms and target_regression:
                target = torch.cat([target[:, :, :p_size], target[:, :, agg_size:agg_size + lr_size]], dim=-1)
            else:
                target = target[:, :, agg_size:agg_size + lr_size]

            if self.debug:
                print(f'--- p {aggregates_p.shape}')
                for i in range(3):
                    print(aggregates_p[i, :, 0])
                print(f'--- d {aggregates_d.shape}')
                for i in range(3):
                    print(aggregates_d[i, :, 0])
                print(f'--- s {source.shape}')
                for i in range(3):
                    print(source[i, :, 0])
                print(f'--- t {target.shape}')
                for i in range(3):
                    print(target[i, :, 0])
            if return_density:
                if single_ts:
                    aggregates = torch.cat([aggregates_p, aggregates_d], dim=-1)
                else:
                    aggregates = torch.stack([aggregates_p, aggregates_d], dim=-2)
            else:
                aggregates = aggregates_p
            if self.debuge:
                print(f'p={aggregates_p.shape}, d={aggregates_d.shape}, res={aggregates.shape}, '
                      f's={source.shape}, t={target.shape}')

            return (aggregates, source), target

        return __split_window

    def to_np_dataset(self, blob, in_len, out_len, with_reverse, step=1):
        class NpDataset(data.Dataset):
            def __init__(self, array): self.array = array

            def __len__(self): return len(self.array)

            def __getitem__(self, i): return self.array[i]

        data_len = max(in_len, out_len)
        out_start = min(in_len, out_len) + self.output_offset
        buffer_size = in_len + out_len
        attempts = blob.shape[0]
        time = blob.shape[1]
        ts_data = []
        for k in np.arange(0, time - buffer_size, step=step):
            for a in range(attempts):
                """ blob = attempt, time, (p, d, lr)"""
                x = blob[a, k:k + data_len, :]
                y = blob[a, k + out_start:k + data_len + out_start, :]
                row = torch.stack([x, y], dim=-1)
                ts_data.append(row)
                if with_reverse:
                    ts_data.append(torch.flip(row, [0]))

        return NpDataset(ts_data)

    def to_dataset(self, agg, in_len, lrs, out_len, with_reverse):
        blob, a_size, lr_size = self.compress_into_single_matrix(agg, lrs)
        return self.to_np_dataset(blob=blob,
                                  in_len=in_len,
                                  out_len=out_len,
                                  with_reverse=with_reverse), a_size, lr_size

    def data_loader(self, agg, lrs, is_train=False, shuffle=True, batch_size=DEFAULT_BATCH):
        dataset, a_size, lr_size = self.to_dataset(agg, in_len, lrs, out_len, with_reverse)

        dl = data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

        return map(self._split_window(
            in_len=in_len,
            out_len=out_len,
            lr_size=lr_size,
            agg_size=a_size), dl)

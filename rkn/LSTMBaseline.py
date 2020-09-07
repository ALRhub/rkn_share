import torch
import numpy as np
from util.TimeDistributed import TimeDistributed
from util.Losses import mse, gaussian_nll
import time as t
from rkn.Encoder import Encoder
from rkn.Decoder import SplitDiagGaussianDecoder
from rkn.RKNLayer import RKNLayer
from util.ConfigDict import ConfigDict
from typing import Tuple
from rkn.RKN import RKN
optim = torch.optim
nn = torch.nn


class LSTMBaseline(RKN):

    def __init__(self, target_dim: int, lod: int, cell_config: ConfigDict, use_cuda_if_available: bool = True):
        """
        TODO: Gradient Clipping?
        :param target_dim:
        :param lod:
        :param cell_config:
        :param use_cuda_if_available:
        """

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")

        self._lod = lod
        self._lsd = 2 * self._lod

        # parameters TODO: Make configurable
        self._enc_out_normalization = "pre"
        self._initial_state_variance = 10.0
        self._learning_rate = 1e-3

        # main model
        # Its not ugly, its pythonic :)
        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(lod, output_normalization=self._enc_out_normalization)
        self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        self._rkn_layer = nn.LSTM(input_size=2 * lod, hidden_size=5 * lod, batch_first=True).to(self._device)

        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        self._dec = TimeDistributed(SplitDiagGaussianDecoder(lod, out_dim=target_dim), num_outputs=2).to(self._device)

        # params and optimizer
        self._params = list(self._enc.parameters())
        self._params += list(self._rkn_layer.parameters())
        self._params += list(self._dec.parameters())


        self._optimizer = optim.Adam(self._params, lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def _train_on_batch(self, obs_batch: torch.Tensor, target_batch: torch.Tensor) -> Tuple[float, float]:
        """Single update step on a batch
        :param obs_batch: batch of observation sequences
        :param target_batch: batch of target sequences
        :return: loss (nll) and metric (rmse)
        """

        self._optimizer.zero_grad()

        w, w_var = self._enc(obs_batch)
        out, _ = self._rkn_layer(torch.cat([w, w_var], dim=-1))
        out_mean, out_var = self._dec(out[..., : 2 * self._lod].contiguous(), out[..., 2 * self._lod:].contiguous())

        loss = gaussian_nll(target_batch, out_mean, out_var)
        loss.backward()
        self._optimizer.step()

        with torch.no_grad():
            metric = mse(target_batch, out_mean)

        return loss.detach().cpu().numpy(), metric.detach().cpu().numpy()

    def eval(self, obs: np.ndarray, targets: np.ndarray, batch_size) -> Tuple[float, float]:
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        scale_factor = 255.0 if obs.dtype == np.uint8 else 1.0

        num_batches = int(obs.shape[0] / batch_size)
        avg_loss = 0.0
        avg_metric = 0.0
        for i in range(num_batches):
            cur_slice = slice(i * batch_size, (i + 1) * batch_size)
            with torch.no_grad():
                torch_obs = torch.from_numpy(obs[cur_slice].astype(np.float32) / scale_factor).to(self._device)
                torch_targets = torch.from_numpy(targets[cur_slice].astype(np.float32)).to(self._device)

                w, w_var = self._enc(torch_obs)
                out, _ = self._rkn_layer(torch.cat([w, w_var], dim=-1))
                out_mean, out_var = self._dec(out[..., : 2 * self._lod].contiguous(), out[..., 2 * self._lod:].contiguous())

                loss = gaussian_nll(torch_targets, out_mean, out_var)
                metric = mse(torch_targets, out_mean)

                avg_loss += loss.detach().cpu().numpy() / num_batches
                avg_metric += metric.detach().cpu().numpy() / num_batches

        return avg_loss, avg_metric
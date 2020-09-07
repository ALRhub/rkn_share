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

optim = torch.optim
nn = torch.nn


class RKN:

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

        self._rkn_layer = RKNLayer(latent_obs_dim=lod, cell_config=cell_config).to(self._device)

        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        self._dec = TimeDistributed(SplitDiagGaussianDecoder(lod, out_dim=target_dim), num_outputs=2).to(self._device)

        # build (default) initial state
        self._initial_mean = torch.zeros(1, self._lsd).to(self._device)
        self._icu = torch.nn.Parameter(self._initial_state_variance * torch.ones(1, self._lod).to(self._device))
        self._icl = torch.nn.Parameter(self._initial_state_variance * torch.ones(1, self._lod).to(self._device))
        self._ics = torch.zeros(1, self._lod).to(self._device)

        # params and optimizer
        self._params = list(self._enc.parameters())
        self._params += list(self._rkn_layer.parameters())
        self._params += list(self._dec.parameters())
        self._params += [self._icu, self._icl]

        self._optimizer = optim.Adam(self._params, lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def _build_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_dec_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_dec_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _train_on_batch(self, obs_batch: torch.Tensor, target_batch: torch.Tensor) -> Tuple[float, float]:
        """Single update step on a batch
        :param obs_batch: batch of observation sequences
        :param target_batch: batch of target sequences
        :return: loss (nll) and metric (rmse)
        """

        self._optimizer.zero_grad()

        w, w_var = self._enc(obs_batch)
        post_mean, post_cov  = self._rkn_layer(w, w_var, self._initial_mean, [self._icu, self._icl, self._ics])
        out_mean, out_var = self._dec(post_mean, torch.cat(post_cov, dim=-1))

        loss = gaussian_nll(target_batch, out_mean, out_var)
        loss.backward()
        self._optimizer.step()

        with torch.no_grad():
            metric = mse(target_batch, out_mean)

        return loss.detach().cpu().numpy(), metric.detach().cpu().numpy()

    def train_step(self, train_obs: np.ndarray, train_targets: np.ndarray, batch_size: int) \
            -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_targets: training targets
        :param batch_size:
        :return: average loss (nll) and  average metric (rmse), execution time
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        scale_factor = 255.0 if train_obs.dtype == np.uint8 else 1.0
        batches_per_epoch = int(train_obs.shape[0] / batch_size)
        rnd_idx = self._shuffle_rng.permutation(train_obs.shape[0])
        avg_nll = avg_mse = 0
        t0 = t.time()

        for j in range(batches_per_epoch):

            batch_idx = rnd_idx[j * batch_size: (j + 1) * batch_size]
            obs_batch = torch.from_numpy(train_obs[batch_idx].astype(np.float32) / scale_factor).to(self._device)
            target_batch = torch.from_numpy(train_targets[batch_idx].astype(np.float32)).to(self._device)

            loss = self._train_on_batch(obs_batch=obs_batch, target_batch=target_batch)
            avg_nll += loss[0] / batches_per_epoch
            avg_mse += loss[1] / batches_per_epoch

        return avg_nll, avg_mse, t.time() - t0

    def eval(self, obs: np.ndarray, targets: np.ndarray, batch_size: int=-1) -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param targets: targets to evaluate on
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        scale_factor = 255.0 if obs.dtype == np.uint8 else 1.0

        batch_size = batch_size if batch_size > 0 else obs.shape[0]
        num_batches = int(obs.shape[0] / batch_size)
        avg_loss = 0.0
        avg_metric = 0.0
        for i in range(num_batches):
            cur_slice = slice(i * batch_size, (i + 1) * batch_size)
            with torch.no_grad():
                torch_obs = torch.from_numpy(obs[cur_slice].astype(np.float32) / scale_factor).to(self._device)
                torch_targets = torch.from_numpy(targets[cur_slice].astype(np.float32)).to(self._device)

                w, w_var = self._enc(torch_obs)
                post_mean, post_cov = self._rkn_layer(w, w_var, self._initial_mean, [self._icu, self._icl, self._ics])
                out_mean, out_var = self._dec(post_mean, torch.cat(post_cov, dim=-1))

                loss = gaussian_nll(torch_targets, out_mean, out_var)
                metric = mse(torch_targets, out_mean)

                avg_loss += loss.detach().cpu().numpy() / num_batches
                avg_metric += metric.detach().cpu().numpy() / num_batches

        return avg_loss, avg_metric

    def train(self, train_obs: np.ndarray, train_targets: np.ndarray, epochs: int, batch_size: int,
              val_obs: np.ndarray = None, val_targets: np.ndarray = None, val_interval: int = 1,
              val_batch_size: int = -1) -> None:
        """
        Train function
        :param train_obs: observations for training
        :param train_targets: targets for training
        :param epochs: epochs to train for
        :param batch_size: batch size for training
        :param val_obs: observations for validation
        :param val_targets: targets for validation
        :param val_interval: validate every <this> iterations
        :param val_batch_size: batch size for validation, to save memory
        """

        """ Train Loop"""

        if val_batch_size == -1:
            val_batch_size = 4 * batch_size

        for i in range(epochs):
            train_ll, train_rmse, time = self.train_step(train_obs, train_targets, batch_size)
            print("Training Iteration {:04d}: NLL: {:.5f}, MSE: {:.5f}, Took {:4f} seconds".format(
                i + 1, train_ll, train_rmse, time))
            if val_obs is not None and val_targets is not None and i % val_interval == 0:
                val_ll, val_rmse = self.eval(val_obs, val_targets, batch_size=val_batch_size)
                print("Validation: NLL: {:.5f}, MSE: {:.5f}".format(val_ll, val_rmse))
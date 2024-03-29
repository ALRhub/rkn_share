import torch
from rkn_cell.RKNCell import RKNCell
from rkn.RKN import RKN
import numpy as np
import os
from util.ConfigDict import ConfigDict
from util.MyLayerNorm2d import MyLayerNorm2d
from rkn.LSTMBaseline import LSTMBaseline
nn = torch.nn


class QLStateEstemRKN(RKN):

    def __init__(self, target_dim: int, lod: int, cell_config: ConfigDict, layer_norm: bool,
                 use_cuda_if_available: bool = True):
        self._layer_norm = layer_norm

        super(QLStateEstemRKN, self).__init__(target_dim, lod, cell_config, use_cuda_if_available)

    def _build_enc_hidden_layers(self):
        layers = []
        # hidden layer 1 => images are grayscale -> only one input channel
        layers.append(nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2, stride=2))
        if self._layer_norm:
            layers.append(MyLayerNorm2d(channels=12))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # hidden layer 2
        layers.append(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=2, padding=1))
        if self._layer_norm:
            layers.append(MyLayerNorm2d(channels=12))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # hidden layer 3
        #TODO: Introducing a bottleneck here.. less striding during convolution?
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=108, out_features=200))
        layers.append(nn.ReLU())
        return nn.ModuleList(layers), 200

    def _build_dec_hidden_layers_mean(self):
        return nn.ModuleList([
            nn.Linear(in_features=2 * self._lod, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64),
            nn.Tanh()
        ]), 64

    def _build_dec_hidden_layers_var(self):
        return nn.ModuleList([
            nn.Linear(in_features=3 * self._lod, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64),
            nn.Tanh()
        ]), 64


if __name__ == "__main__":

    latent_obs_dim = 100

    cell_conf = RKNCell.get_default_config()
    cell_conf.num_basis = 15
    cell_conf.bandwidth = 3
    cell_conf.never_invalid = True
    cell_conf.trans_net_hidden_units = [128, 128]
    cell_conf.trans_net_hidden_activation = "Tanh"
    cell_conf.trans_covar = 0.1
    cell_conf.finalize_modifying()

    rkn = QLStateEstemRKN(8, latent_obs_dim, cell_conf, layer_norm=True)

    # Data
    # to get the data, run experiments/quad_link/data/NLinkPendulum once, put the npz somewhere and change path
    # here. The "simulator" needed for the quad-link can be found in the old tensorflow code
    # (https://github.com/LCAS/RKN/tree/master/n_link_sim)
    data_path = "."
    data = dict(np.load(os.path.join(data_path, "ql.npz")))
    train_obs = data["train_obs"]
    train_targets = data["train_targets"]
    test_obs = data["test_obs"]
    test_targets = data["test_targets"]

    train_obs = np.ascontiguousarray(np.transpose(train_obs, [0, 1, 4, 2, 3]))
    test_obs = np.ascontiguousarray(np.transpose(test_obs, [0, 1, 4, 2, 3]))
    train_targets = train_targets.astype(np.float32)
    test_targets = test_targets.astype(np.float32)

    batch_size = 50
    epochs = 500

    rkn.train(train_obs, train_targets, epochs, batch_size, val_obs=test_obs, val_targets=test_targets)

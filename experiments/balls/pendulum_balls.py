import torch
from rkn_cell.RKNCell import RKNCell
from rkn.RKN import RKN
import numpy as np
import os
from util.ConfigDict import ConfigDict
from util.MyLayerNorm2d import MyLayerNorm2d

nn = torch.nn

class BallTrackRKN(RKN):

    def __init__(self, target_dim: int, lod: int, cell_config: ConfigDict, use_cuda_if_available: bool = True):
        super(BallTrackRKN, self).__init__(target_dim, lod, cell_config, use_cuda_if_available)

    def _build_enc_hidden_layers(self):
        return nn.ModuleList([
            # hidden layer 1
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, padding=2),
            MyLayerNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # hidden layer 2
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=2, padding=1),
            MyLayerNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # hidden layer 3
            nn.Flatten(),
            nn.Linear(in_features=192, out_features=30),
            nn.ReLU()
        ]), 30

    def _build_dec_hidden_layers_mean(self):
        return nn.ModuleList([
            nn.Linear(in_features=2 * self._lod, out_features=30),
            nn.Tanh()
        ]), 30

    def _build_dec_hidden_layers_var(self):
        return nn.ModuleList([
            nn.Linear(in_features=3 * self._lod, out_features=30),
            nn.Tanh()
        ]), 30

if __name__ == "__main__":

    latent_obs_dim = 15

    cell_conf = RKNCell.get_default_config()
    cell_conf.num_basis = 15
    cell_conf.bandwidth = 3
    cell_conf.never_invalid = True
    cell_conf.trans_net_hidden_units = []
    cell_conf.trans_net_hidden_activation = "tanh"
    cell_conf.trans_covar = 0.1
    cell_conf.finalize_modifying()

    rkn = BallTrackRKN(2, latent_obs_dim, cell_conf)

    """Data"""
    # to get the data, run experiments/balls/data/PendulumBalls  once, put the npz somewhere and change path here
    data_path = "."
    data = dict(np.load(os.path.join(data_path, "pend_balls_first_clean.npz")))
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




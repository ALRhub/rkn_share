import torch
import numpy as np
import os
from rkn_cell.RKNCell import var_activation, RKNCell
from util.Losses import mse, gaussian_nll
import time as t
nn = torch.nn
optim = torch.optim

""" This tutorial is only to showcase how to use the RKNCell as a standalone module for you own research. Here
we (almost) rebuild the RKN as described in the ICML paper. For an efficient, configurable and reusable version of 
exactly that architecture see also the "rkn" folder.
Also, the Hyperparameters here are not very well tuned so do not be puzzled if it works not that will ;-), the
 pendulum and quadlink experiments work well"""

"""Configure for GPU usage"""
use_gpu_if_available = True
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu_if_available else "cpu")

"""Load the data (you can generate data yourself using data experiments/balls/data/PendulumBalls.py) """
data_path = "."
data = dict(np.load(os.path.join(data_path, "pend_balls_first_clean.npz")))
train_obs = data["train_obs"]
train_targets = data["train_targets"]
test_obs = data["test_obs"]
test_targets = data["test_targets"]
train_obs = np.transpose(train_obs, [0, 1, 4, 2, 3])  # adapt to pytorch image format


""" Build the other parts of the model, as we want to rebuild the original RKN here, we just need an encoder and 
decoder """
class MyEncoder(nn.Module):

    def __init__(self, lod):
        super(MyEncoder, self).__init__()
        self._hidden_layers = self._build_hidden_layers()

        self._mean_layer = nn.Linear(in_features=30, out_features=lod)
        self._log_var_layer = nn.Linear(in_features=30, out_features=lod)

    def _build_hidden_layers(self):
        return nn.ModuleList([
            # hidden layer 1
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # hidden layer 2
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # hidden layer 3
            nn.Flatten(),
            nn.Linear(in_features=192, out_features=30),
            nn.ReLU()
        ])

    def forward(self, obs):
        h = obs
        for layer in self._hidden_layers:
            h = layer(h)
        h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)
        mean = self._mean_layer(h)
        log_var = self._log_var_layer(h)
        var = var_activation(log_var)
        return mean, var


class MyDecoder(nn.Module):

    def __init__(self, lod, out_dim):
        super(MyDecoder, self).__init__()
        self._latent_obs_dim = lod
        self._out_dim = out_dim

        self._hidden_layers_mean = self._build_hidden_layers_mean()
        self._hidden_layers_var = self._build_hidden_layers_var()
        self._out_layer_mean = nn.Linear(in_features=30, out_features=out_dim)
        self._out_layer_var = nn.Linear(in_features=45, out_features=out_dim)

    def _build_hidden_layers_mean(self):
        return nn.ModuleList([
            nn.Linear(in_features=2 * self._latent_obs_dim, out_features=30),
            nn.Tanh()
        ])

    def _build_hidden_layers_var(self):
        return nn.ModuleList([
            nn.Linear(in_features=3 * self._latent_obs_dim, out_features=30),
            nn.Tanh()
        ])

    def forward(self, latent_mean, latent_cov):
        h_mean = latent_mean
        for layer in self._hidden_layers_mean:
            h_mean = layer(h_mean)
        mean = self._out_layer_mean(h_mean)

        h_var = latent_cov
        for layer in self._hidden_layers_var:
            h_var = layer(h_var)
        log_var = self._out_layer_var(latent_cov)
        var = var_activation(log_var)
        return mean, var

enc = MyEncoder(lod=15).to(device)
dec = MyDecoder(lod=15, out_dim=2).to(device)

initial_mean = torch.zeros(1, 30).to(device)
initial_cov = [10.0 * torch.ones(1, 15).to(device),
               10.0 * torch.ones(1, 15).to(device),
               torch.zeros(1, 15).to(device)]

"""Configure and Build the actual RKN Cell"""
cell_conf = RKNCell.get_default_config()
cell_conf.num_basis = 15
cell_conf.bandwidth = 3
cell_conf.never_invalid = True
cell_conf.trans_net_hidden_units = []
cell_conf.trans_net_hidden_activation = "tanh"
cell_conf.trans_covar = 0.1
cell_conf.finalize_modifying()
rkn_cell = RKNCell(latent_obs_dim=15, config=cell_conf).to(device)

"""Build Optimizer"""
params = list(enc.parameters()) + list(rkn_cell.parameters()) + list(dec.parameters())
optimizer = optim.Adam(params, lr=1e-3)


"""Training"""
batch_size = 50
epochs = 100
batches_per_epoch = int(train_obs.shape[0] / batch_size)
np.random.seed(0)



"""Write some train loop"""
def _train_on_batch(obs_batch, target_batch):
    seq_length = obs_batch.shape[1]
    optimizer.zero_grad()

    out_means = []
    out_vars = []

    # initialize prior belief
    prior_mean = initial_mean
    prior_cov = initial_cov

    for t in range(seq_length):
        # encode observation
        w, w_var = enc(obs_batch[:, t])

        # filter step
        post_mean, post_cov, next_prior_mean, next_prior_cov = rkn_cell(prior_mean, prior_cov, w, w_var)

        # decode posterior belief
        out_mean, out_var = dec(post_mean, torch.cat(post_cov, dim=-1))
        out_means.append(out_mean)
        out_vars.append(out_var)

        # forward cell state
        prior_mean = next_prior_mean
        prior_cov = next_prior_cov

    # stack outputs
    out_mean = torch.stack(out_means, 1)
    out_var = torch.stack(out_vars, 1)

    # compute loss, gradient and update
    loss = gaussian_nll(target_batch, out_mean, out_var)
    loss.backward()
    optimizer.step()

    # rmse as evaluation metric
    with torch.no_grad():
        metric = mse(target_batch, out_mean)

    return loss, metric

""" iterate train loop """

for i in range(epochs):
    rnd_idx = np.random.permutation(train_obs.shape[0])
    avg_nll = avg_rmse = 0
    t0 = t.time()
    for j in range(batches_per_epoch):
        batch_idx = rnd_idx[j * batch_size: (j + 1) * batch_size]
        obs_batch = train_obs[batch_idx]
        target_batch = train_targets[batch_idx]

        loss = _train_on_batch(obs_batch=torch.from_numpy(obs_batch.astype(np.float32)/ 255.0).to(device),
                               target_batch=torch.from_numpy(target_batch.astype(np.float32)).to(device))
        avg_nll += loss[0] / batches_per_epoch
        avg_rmse += loss[1] / batches_per_epoch
    print(i, avg_nll, avg_rmse, t.time() - t0)

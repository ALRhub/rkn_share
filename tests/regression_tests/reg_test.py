import torch
import numpy as np
from rkn.RKNLayer import RKNLayer
from rkn_cell.RKNCell import RKNCell
from tests.regression_tests.reference_implementation.TF_RKNLayer import TFRKNLayer
import os

"""Regression Test of the pytorch cell against the original tensorflow cell"""

os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""test params"""
lod = 100
seq_length = 75
batch_size = 50

"""data"""
data_rng = np.random.RandomState(0)

obs_mean = data_rng.randn(batch_size, seq_length, lod).astype(np.float32)
obs_var = np.exp(data_rng.randn(batch_size, seq_length, lod)).astype(np.float32)

init_mean = np.zeros([batch_size, 2 * lod], np.float32)
init_cov = [np.ones([batch_size, lod], np.float32),
            np.ones([batch_size, lod], np.float32),
            np.zeros([batch_size, lod], np.float32)]

obs_valid = data_rng.uniform(low=0, high=1, size=[batch_size, seq_length]) > 0.5

"""torch test"""

torch_config = RKNCell.get_default_config()
torch_config.never_invalid = False
torch_config.finalize_modifying()
rkn_layer = RKNLayer(lod, torch_config)
torch_obs_mean = torch.from_numpy(obs_mean)
torch_obs_var = torch.from_numpy(obs_var)
torch_init_mean = torch.from_numpy(init_mean)
torch_init_cov = [torch.from_numpy(x) for x in init_cov]
torch_obs_valid = torch.from_numpy(obs_valid)

out = rkn_layer(torch_obs_mean, torch_obs_var, torch_init_mean, torch_init_cov, torch_obs_valid)
post_mean_torch = out[0].detach().numpy()
post_cov_torch = [x.detach().numpy() for x in out[1]]
prior_mean_torch = out[2].detach().numpy()
prior_cov_torch = [x.detach().numpy() for x in out[3]]

"""tf test"""
tf_config = TFRKNLayer.get_default_config()
tf_config.never_invalid = False
tf_config.finalize_modifying()

tf_rkn_layer = TFRKNLayer(lod, tf_config)

out = tf_rkn_layer(obs_mean, obs_var, np.expand_dims(obs_valid, -1))
post_mean_tf = out[0].numpy()
post_cov_tf = [x.numpy() for x in out[1]]
prior_mean_tf = out[2].numpy()
prior_cov_tf = [x.numpy() for x in out[3]]

"""test"""
print("Max Differences")
print("Posterior Mean")
print(np.max(np.abs(post_mean_tf - post_mean_torch)))
print("Posterior Cov")
print(np.max(np.abs(post_cov_tf[0] - post_cov_torch[0])))
print(np.max(np.abs(post_cov_tf[1] - post_cov_torch[1])))
print(np.max(np.abs(post_cov_tf[2] - post_cov_torch[2])))

print("Prior Mean")
print(np.max(np.abs(prior_mean_tf - prior_mean_torch)))
print("Prior Cov")
print(np.max(np.abs(prior_cov_tf[0] - prior_cov_torch[0])))
print(np.max(np.abs(prior_cov_tf[1] - prior_cov_torch[1])))
print(np.max(np.abs(prior_cov_tf[2] - prior_cov_torch[2])))

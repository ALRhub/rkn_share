import torch
import numpy as np
from rkn.RKNLayer import RKNLayer
from rkn_cell.RKNCell import RKNCell
import os

optim = torch.optim


#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
torch_config.never_invalid = True
torch_config.finalize_modifying()
rkn_layer = RKNLayer(lod, torch_config)


opt = optim.Adam(rkn_layer.parameters(), lr=1e-3)

torch_obs_mean = torch.from_numpy(obs_mean)
torch_obs_var = torch.from_numpy(obs_var)
torch_init_mean = torch.from_numpy(init_mean)
torch_init_cov = [torch.from_numpy(x) for x in init_cov]
torch_obs_valid = torch.from_numpy(obs_valid)


opt.zero_grad()

out = rkn_layer(torch_obs_mean, torch_obs_var, torch_init_mean, torch_init_cov, torch_obs_valid)

loss = out[0].square().mean()
loss.backward()
opt.step()

post_mean_torch = out[0].detach().numpy()
post_cov_torch = [x.detach().numpy() for x in out[1]]
prior_mean_torch = out[2].detach().numpy()
prior_cov_torch = [x.detach().numpy() for x in out[3]]

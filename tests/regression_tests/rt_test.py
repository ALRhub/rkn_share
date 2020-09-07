import torch
import numpy as np
from rkn.RKNLayer import RKNLayer
from rkn_cell.RKNCell import RKNCell
import time as t
#from experiments.regression_tests.reference_implementation.TF_RKNLayer import TFRKNLayer

"""Runtime tests of the pytorch cell vs the original tensorflow cell. Make sure only one is executed at any time.
Tensorflow on Torch at the same time on the same GPU is a bad idea"""

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

"""torch test"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_config = RKNCell.get_default_config()
torch_config.never_invalid = True
torch_config.finalize_modifying()
rkn = RKNLayer(lod, torch_config).to(device)

for i in range(10):
    t0 = t.time()
    torch_obs_mean = torch.from_numpy(obs_mean).to(device)
    torch_obs_var = torch.from_numpy(obs_var).to(device)
    torch_init_mean = torch.from_numpy(init_mean).to(device)
    torch_init_cov = [torch.from_numpy(x).to(device) for x in init_cov]
    torch_obs_valid = torch.from_numpy(np.ones([batch_size, seq_length], dtype=bool)).to(device)
    post_mean, post_cov, prior_mean, prior_cov = \
        rkn(torch_obs_mean, torch_obs_var, torch_init_mean, torch_init_cov) #, torch_obs_valid)
    print(t.time() - t0)


"""tf test"""
#tf_config = TFRKNLayer.get_default_config()
#tf_config.never_invalid = False
#tf_config.finalize_modifying()

#tf_rkn_layer = TFRKNLayer(lod, tf_config)


#tf_obs_mean = obs_mean.astype(np.float32)
#tf_obs_var = obs_var.astype(np.float32)
#tf_obs_valid = np.ones([batch_size, seq_length, 1], dtype=np.float32)

#for i in range(10):
#    t0 = t.time()
#    out = tf_rkn_layer(tf_obs_mean, tf_obs_var, tf_obs_valid)
#    print(t.time() - t0)
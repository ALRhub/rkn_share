from tests.spring_damper.SpringDamper import SpringDamper
from tests.spring_damper.KalmanSmoother import KalmanSmoother
import numpy as np
import matplotlib.pyplot as plt
from rkn.RKNLayer import RKNLayer
from rkn_cell.RKNCell import RKNCell
import torch

"""Simple test to see if the cell acts like a kalman filter if all assumptions are meet"""


np.random.seed(0)

sd = SpringDamper(sigma_obs=1.2, sigma_trans=0.1, dt=0.2, m=1.0, b=1.0)

init_state = np.concatenate([np.random.uniform(low=-0.2, high=0.2, size=2),
                             np.random.uniform(low=-0.2, high=0.2, size=2)], 0)
obs, states = sd.run(init_state, 100)

print(sd.transition_matrix)

smoother = KalmanSmoother(transition_matrix=sd.transition_matrix,
                          observation_matrix=sd.observation_matrix,
                          transition_noise=sd.transition_noise_covar,
                          observation_noise=sd.observation_noise_covar)

config = RKNCell.get_default_config()
config.trans_covar = sd.transition_noise_covar[0, 0]
config.bandwidth = 0

config.finalize_modifying()

rkn_layer = RKNLayer(2, config)

initial_mean = np.zeros(4)
initial_covar = np.eye(4)

ref_post_means, ref_post_covars, ref_prior_means, ref_prior_covars, mbf_dict = \
    smoother.forward_pass(obs, initial_mean=np.expand_dims(initial_mean, -1), initial_covar=initial_covar)

rkn_post_mean, rkn_post_cov, rkn_prior_mean, rkn_prior_cov = \
    rkn_layer(torch.from_numpy(obs[None, :, :].astype(np.float32)),
              sd.observation_noise_covar[0, 0] * torch.ones([1, 101, 2]),
              torch.zeros(1, 4),
              [torch.ones(1, 2), torch.ones(1, 2), torch.zeros(1, 2)])

rkn_post_mean = rkn_post_mean.detach().numpy()
rkn_post_cov = [x.detach().numpy() for x in rkn_post_cov]
rkn_prior_mean = rkn_prior_mean.detach().numpy()
rkn_prior_cov = [x.detach().numpy() for x in rkn_prior_cov]


def construct_full_covar(var_list):
    seq_length,  od = var_list[0].shape[1:]
    cu = np.expand_dims(np.eye(od), 0) * np.expand_dims(np.squeeze(var_list[0]), -1)
    cl = np.expand_dims(np.eye(od), 0) * np.expand_dims(np.squeeze(var_list[1]), -1)
    cs = np.expand_dims(np.eye(od), 0) * np.expand_dims(np.squeeze(var_list[2]), -1)
    c = np.concatenate([np.concatenate([cu, cs], axis=-2),
                        np.concatenate([cs, cl], axis=-2)], axis=-1)
    return c

print("Posterior")
print(np.max(np.abs(np.squeeze(rkn_post_mean) - np.squeeze(ref_post_means))))
print(np.max(np.abs(np.squeeze(construct_full_covar(rkn_post_cov)) - np.squeeze(ref_post_covars))))

print("Prior")
print(np.max(np.abs(np.squeeze(rkn_prior_mean)[:-1] - np.squeeze(ref_prior_means)[1:])))
print(np.max(np.abs(np.squeeze(construct_full_covar(rkn_prior_cov))[:-1] - np.squeeze(ref_prior_covars)[1:])))

plt.title("Posterior")
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(states[:, i], c="green")
    plt.plot(ref_post_means[:, i], c="blue")
    plt.plot(np.squeeze(rkn_post_mean)[:, i], c="red")
    if i < 2:
        plt.scatter(np.arange(0, len(obs), 1), obs[:, i], c="green", marker=".")
    if i == 0:
        plt.legend(["GT", "Kalman Filter", "RKN"])

plt.figure()
plt.title("Prior")
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(states[:, i], c="green")
    plt.plot(ref_prior_means[1:, i], c="blue")
    plt.plot(np.squeeze(rkn_prior_mean)[:-1, i], c="red")
    if i < 2:
        plt.scatter(np.arange(0, len(obs), 1), obs[:, i], c="green", marker=".")
    if i == 0:
        plt.legend(["GT", "Kalman Filter", "RKN"])

plt.show()
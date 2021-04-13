import torch
import numpy as np
from util.ConfigDict import ConfigDict
from typing import Iterable, Tuple, List, Union
nn = torch.nn


def bmv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Batched Matrix Vector Product"""
    return torch.bmm(mat, vec[..., None])[..., 0]


def dadat(a: torch.Tensor, diag_mat: torch.Tensor) -> torch.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * A^T) where A is a batch of square matrices and
    diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch of square matrices,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    return bmv(a.square(), diag_mat)


def dadbt(a: torch.Tensor, diag_mat: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * B^T) where A and B are batches of square matrices and
     diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch square matrices
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :param b: batch of square matrices
    :returns diagonal entries of  A * diag_mat * B^T"""
    return bmv(a * b, diag_mat)


def var_activation(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.log(torch.exp(x) + 1.0)


def var_activation_inverse(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(np.exp(x) - 1.0)


class RKNCell(nn.Module):

    @staticmethod
    def get_default_config() -> ConfigDict:
        config = ConfigDict(
            num_basis=15,
            bandwidth=3,
            trans_net_hidden_units=[64, 64],
            trans_net_hidden_activation="Tanh",
            learn_trans_covar=True,
            trans_covar=0.1,
            learn_initial_state_covar=False,
            initial_state_covar=1,
            never_invalid=True
        )
        config.finalize_adding()
        return config

    def __init__(self, latent_obs_dim: int, config: ConfigDict, dtype: torch.dtype = torch.float32):
        """
        RKN Cell (mostly) as described in the original RKN paper
        :param latent_obs_dim: latent observation dimension
        :param config: config dict object, for configuring the cell
        :param dtype: datatype
        """
        super(RKNCell, self).__init__()
        self._lod = latent_obs_dim
        self._lsd = 2 * self._lod

        self.c = config
        self._dtype = dtype

        self._build_transition_model()

    def _compute_band_util(self,
                           lod: int,
                           bandwidth: int):
        self._num_entries = lod + 2 * np.sum(np.arange(lod - bandwidth, lod))
        np_mask = np.ones([lod, lod], dtype=np.float32)
        np_mask = np.triu(np_mask, -bandwidth) * np.tril(np_mask, bandwidth)
        mask = torch.tensor(np_mask, dtype=torch.bool)
        idx = torch.where(mask == 1)
        diag_idx = torch.where(idx[0] == idx[1])

        self.register_buffer("_idx0", idx[0], persistent=False)
        self.register_buffer("_idx1", idx[1], persistent=False)
        self.register_buffer("_diag_idx", diag_idx[0], persistent=False)

    def _unflatten_tm(self,
                      tm_flat: torch.Tensor) -> torch.Tensor:
        tm = torch.zeros(tm_flat.shape[0], self._lod, self._lod, device=tm_flat.device)
        tm[:, self._idx0, self._idx1] = tm_flat
        return tm

    def forward(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor],
                obs: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Iterable[torch.Tensor], torch.Tensor, Iterable[torch.Tensor]]:
        """
        forward pass trough the cell. For proper recurrent model feed back outputs 3 and 4 (next prior belief at next
        time step

        :param prior_mean: prior mean at time t
        :param prior_cov: prior covariance at time t
        :param obs: observation at time t
        :param obs_var: observation variance at time t
        :param obs_valid: flag indicating whether observation at time t valid
        :return: posterior mean at time t, posterior covariance at time t
                 prior mean at time t + 1, prior covariance time t + 1
        """
        if self.c.never_invalid:
            post_mean, post_cov = self._update(prior_mean, prior_cov, obs, obs_var)
        else:
            assert obs_valid is not None
            post_mean, post_cov = self._masked_update(prior_mean, prior_cov, obs, obs_var, obs_valid)

        next_prior_mean, next_prior_cov = self._predict(post_mean, post_cov)

        return post_mean, post_cov, next_prior_mean, next_prior_cov

    def _build_coefficient_net(self, num_hidden: Iterable[int], activation: str) -> torch.nn.Sequential:
        """
        builds the network computing the coefficients from the posterior mean. Currently only fully connected
        neural networks with same activation across all hidden layers supported
        TODO: Allow more flexible architectures
        :param num_hidden: number of hidden uints per layer
        :param activation: hidden activation
        :return: coefficient network
        """
        layers = []
        prev_dim = self._lsd
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self.c.num_basis))
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers).to(dtype=self._dtype)

    def _build_transition_model(self) -> None:
        """
        Builds the basis functions for transition model and the nosie
        :return:
        """
        # build state independent basis matrices
        self._compute_band_util(lod=self._lod, bandwidth=self.c.bandwidth)

        self._tm_11_basis = nn.Parameter(torch.zeros(self.c.num_basis, self._num_entries))

        tm_12_init = torch.zeros(self.c.num_basis, self._num_entries)
        tm_12_init[:, self._diag_idx] += 0.2 * torch.ones(self._lod)
        self._tm_12_basis = nn.Parameter(tm_12_init)

        tm_21_init = torch.zeros(self.c.num_basis, self._num_entries)
        tm_21_init[:, self._diag_idx] -= 0.2 * torch.ones(self._lod)
        self._tm_21_basis = nn.Parameter(tm_21_init)

        self._tm_22_basis = nn.Parameter(torch.zeros(self.c.num_basis, self._num_entries))
        self._transition_matrices_raw = [self._tm_11_basis, self._tm_12_basis, self._tm_21_basis, self._tm_22_basis]

        self._coefficient_net = self._build_coefficient_net(self.c.trans_net_hidden_units,
                                                            self.c.trans_net_hidden_activation)

        init_trans_cov = var_activation_inverse(self.c.trans_covar)
        # TODO: This is currently a different noise for each dim, not like in original paper (and acrkn)
        self._log_transition_noise = \
            nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd, dtype=self._dtype), init_trans_cov))

    def get_transition_model(self, post_mean: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the locally-linear transition model given the current posterior mean
        :param post_mean: current posterior mean
        :return: transition matrices (4 Blocks), transition covariance (vector of size lsd)
        """
        # prepare transition model
        coefficients = torch.reshape(self._coefficient_net(post_mean), [-1, self.c.num_basis, 1])

        tm11_flat = (coefficients * self._tm_11_basis).sum(dim=1)
        tm11_flat[:, self._diag_idx] += 1.0

        tm12_flat = (coefficients * self._tm_12_basis).sum(dim=1)
        tm21_flat = (coefficients * self._tm_21_basis).sum(dim=1)

        tm22_flat = (coefficients * self._tm_22_basis).sum(dim=1)
        tm22_flat[:, self._diag_idx] += 1.0

        tm11, tm12, tm21, tm22 = [self._unflatten_tm(x) for x in [tm11_flat, tm12_flat, tm21_flat, tm22_flat]]
        trans_cov = var_activation(self._log_transition_noise)

        return [tm11, tm12, tm21, tm22], trans_cov


    def _update(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor],
                obs_mean: torch.Tensor, obs_var: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Performs update step
        :param prior_mean: current prior state mean
        :param prior_cov: current prior state covariance
        :param obs_mean: current observation mean
        :param obs_var: current covariance mean
        :return: current posterior state and covariance
        """
        cov_u, cov_l, cov_s = prior_cov

        # compute kalman gain (eq 2 and 3 in paper)
        denominator = cov_u + obs_var
        q_upper = cov_u / denominator
        q_lower = cov_s / denominator

        # update mean (eq 4 in paper)
        residual = obs_mean - prior_mean[:, :self._lod]
        new_mean = prior_mean + torch.cat([q_upper * residual, q_lower * residual], -1)

        # update covariance (eq 5 -7 in paper)
        covar_factor = 1 - q_upper
        new_covar_upper = covar_factor * cov_u
        new_covar_lower = cov_l - q_lower * cov_s
        new_covar_side = covar_factor * cov_s

        return new_mean, [new_covar_upper, new_covar_lower, new_covar_side]

    def _masked_update(self,
                       prior_mean: torch.Tensor, prior_covar: Iterable[torch.Tensor],
                       obs_mean: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Ensures update only happens if observation is valid
        :param prior_mean: current prior state mean
        :param prior_covar: current prior state covariance
        :param obs_mean: current observation mean
        :param obs_var: current observation covariance
        :param obs_valid: indicating if observation is valid
        :return: current posterior state mean and covariance
        """
        obs_valid = obs_valid[..., None]
        update_mean, update_covar = self._update(prior_mean, prior_covar, obs_mean, obs_var)

        masked_mean = update_mean.where(obs_valid, prior_mean)
        masked_covar_upper = update_covar[0].where(obs_valid, prior_covar[0])
        masked_covar_lower = update_covar[1].where(obs_valid, prior_covar[1])
        masked_covar_side = update_covar[2].where(obs_valid, prior_covar[2])

        return masked_mean, [masked_covar_upper, masked_covar_lower, masked_covar_side]

    def _predict(self, post_mean: torch.Tensor, post_covar: List[torch.Tensor]) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Performs prediction step
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :return: current prior state mean and covariance
        """
        # compute state dependent transition matrix
        [tm11, tm12, tm21, tm22], trans_covar = self.get_transition_model(post_mean)

        # prepare transition noise
        trans_covar_upper = trans_covar[..., :self._lod]
        trans_covar_lower = trans_covar[..., self._lod:]

        # predict next prior mean
        mu = post_mean[:, :self._lod]
        ml = post_mean[:, self._lod:]

        nmu = bmv(tm11, mu) + bmv(tm12, ml)
        nml = bmv(tm21, mu) + bmv(tm22, ml)

        # predict next prior covariance (eq 10 - 12 in paper supplement)
        cu, cl, cs = post_covar
        ncu = dadat(tm11, cu) + 2.0 * dadbt(tm11, cs, tm12) + dadat(tm12, cl) + trans_covar_upper
        ncl = dadat(tm21, cu) + 2.0 * dadbt(tm21, cs, tm22) + dadat(tm22, cl) + trans_covar_lower
        ncs = dadbt(tm21, cu, tm11) + dadbt(tm22, cs, tm11) + dadbt(tm21, cs, tm12) + dadbt(tm22, cl, tm12)
        return torch.cat([nmu, nml], dim=-1), [ncu, ncl, ncs]



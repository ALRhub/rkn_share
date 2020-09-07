import torch
import numpy as np
from util.ConfigDict import ConfigDict
from typing import Iterable, Tuple, List
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


def elup1(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.exp(x).where(x < 0.0, x + 1.0)


def elup1_inv(x: torch.Tensor) -> torch.Tensor:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(x) if x < 1.0 else (x - 1.0)


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

    @property
    def _device(self):
        return self._tm_11_full.device

 #   @torch.jit.script_method
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
        self._band_mask = self._band_mask.to(self._device)

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
        np_mask = np.ones([self._lod, self._lod], dtype=np.float32)
        np_mask = np.triu(np_mask,  -self.c.bandwidth) * np.tril(np_mask, self.c.bandwidth)
        self._band_mask = torch.from_numpy(np.expand_dims(np_mask, 0))

        self._tm_11_full = nn.Parameter(torch.zeros(self.c.num_basis, self._lod, self._lod, dtype=self._dtype))
        self._tm_12_full = \
            nn.Parameter(0.2 * torch.eye(self._lod, dtype=self._dtype)[None, :, :].repeat(self.c.num_basis, 1, 1))
        self._tm_21_full =\
            nn.Parameter(-0.2 * torch.eye(self._lod, dtype=self._dtype)[None, :, :].repeat(self.c.num_basis, 1, 1))
        self._tm_22_full = nn.Parameter(torch.zeros(self.c.num_basis, self._lod, self._lod, dtype=self._dtype))
        self._transition_matrices_raw = [self._tm_11_full, self._tm_12_full, self._tm_21_full, self._tm_22_full]

        self._coefficient_net = self._build_coefficient_net(self.c.trans_net_hidden_units,
                                                            self.c.trans_net_hidden_activation)

        init_trans_cov = elup1_inv(self.c.trans_covar)
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
        coefficients = torch.reshape(self._coefficient_net(post_mean), [-1, self.c.num_basis, 1, 1])

        tm11_full, tm12_full, tm21_full, tm22_full = [x[None, :, :, :] for x in self._transition_matrices_raw]

        tm11_full = (coefficients * tm11_full).sum(dim=1)
        tm11 = tm11_full * self._band_mask
        tm11 += torch.eye(self._lod, device=self._device)[None, :, :]

        tm12_full = (coefficients * tm12_full).sum(dim=1)
        tm12 = tm12_full * self._band_mask

        tm21_full = (coefficients * tm21_full).sum(dim=1)
        tm21 = tm21_full * self._band_mask

        tm22_full = (coefficients * tm22_full).sum(dim=1)
        tm22 = tm22_full * self._band_mask
        tm22 += torch.eye(self._lod, device=self._device)[None, :, :]

        trans_cov = elup1(self._log_transition_noise)

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



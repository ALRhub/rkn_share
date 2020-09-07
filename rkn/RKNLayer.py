import torch
from rkn_cell.RKNCell import RKNCell
nn = torch.nn


class RKNLayer(nn.Module):

    def __init__(self, latent_obs_dim, cell_config, dtype=torch.float32):
        super().__init__()
        self._lod = latent_obs_dim
        self._lsd = 2 * latent_obs_dim
        self._cell = RKNCell(latent_obs_dim, cell_config, dtype)

    def forward(self, latent_obs, obs_vars, initial_mean, initial_cov, obs_valid=None):
        """
        This currently only returns the posteriors. If you also need the priors uncomment the corresponding parts

        :param latent_obs: latent observations
        :param obs_vars: uncertainty estimate in latent observations
        :param initial_mean: mean of initial belief
        :param initial_cov: covariance of initial belief (as 3 vectors)
        :param obs_valid: flags indicating which observations are valid, which are not
        """

        # tif you need a version that also returns the prior uncomment the respective parts below
        # prepare list for return

        #prior_mean_list = []
        #prior_cov_list = [[], [], []]

        post_mean_list = []
        post_cov_list = [[], [], []]


        # initialize prior
        prior_mean, prior_cov = initial_mean, initial_cov

        # actual computation
        for i in range(latent_obs.shape[1]):
            cur_obs_valid = obs_valid[:, i] if obs_valid is not None else None
            post_mean, post_cov, next_prior_mean, next_prior_cov = \
                self._cell(prior_mean, prior_cov, latent_obs[:, i], obs_vars[:, i], cur_obs_valid)

            post_mean_list.append(post_mean)
            [post_cov_list[i].append(post_cov[i]) for i in range(3)]
            #prior_mean_list.append(next_prior_mean)
            #[prior_cov_list[i].append(next_prior_cov[i]) for i in range(3)]

            prior_mean = next_prior_mean
            prior_cov = next_prior_cov

        # stack results
        #prior_means = torch.stack(prior_mean_list, 1)
        #prior_covs = [torch.stack(x, 1) for x in prior_cov_list]

        post_means = torch.stack(post_mean_list, 1)
        post_covs = [torch.stack(x, 1) for x in post_cov_list]

        return post_means, post_covs #, prior_means, prior_covs

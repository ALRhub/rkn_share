import numpy as np


class KalmanSmoother:

    def __init__(self,
                 transition_matrix,
                 observation_matrix,
                 transition_noise,
                 observation_noise):
        self._transition_matrix = transition_matrix
        self._observation_matrix = observation_matrix
        self._transition_noise = transition_noise
        self._observation_noise = observation_noise
        self._state_dim = transition_matrix.shape[0]
        self._obs_dim = observation_matrix.shape[0]

    def forward_pass(self, obs, initial_mean=None, initial_covar=None):

        def update(prior_mean, prior_covar, obs):

            kalman_gain_nominator = prior_covar @ self._observation_matrix.T
            inv_residual_covar = np.linalg.inv(self._observation_matrix @ kalman_gain_nominator + self._observation_noise)
            kalman_gain = kalman_gain_nominator @ inv_residual_covar

            residual = obs - self._observation_matrix @ prior_mean
            post_mean = prior_mean + kalman_gain @ residual

            covar_updt = np.eye(self._state_dim) - kalman_gain @ self._observation_matrix
            post_covar = covar_updt @ prior_covar
            return post_mean, post_covar, kalman_gain, residual, inv_residual_covar, covar_updt

        def predict(post_mean, post_covar):
            next_prior_mean = self._transition_matrix @ post_mean
            next_prior_covar = self._transition_matrix @ post_covar @ self._transition_matrix.T + self._transition_noise
            return next_prior_mean, next_prior_covar

        seq_len = len(obs)
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, -1)

        prior_means = np.zeros([seq_len, self._state_dim, 1])
        prior_covars = np.zeros([seq_len, self._state_dim, self._state_dim])

        post_means = np.zeros([seq_len, self._state_dim, 1])
        post_covars = np.zeros([seq_len, self._state_dim, self._state_dim])

        residuals = np.zeros([seq_len, self._obs_dim, 1])
        inv_residual_covars = np.zeros([seq_len, self._obs_dim, self._obs_dim])
        kalman_gains = np.zeros([seq_len, self._state_dim, self._obs_dim])
        covar_updts = np.zeros([seq_len, self._state_dim, self._state_dim])

        initial_mean = np.zeros([self._state_dim, 1]) if initial_mean is None else initial_mean
        initial_covar = 5 * np.eye(self._state_dim) if initial_covar is None else initial_covar

        for i in range(seq_len):
            if i == 0:
                # initial
                prior_means[i], prior_covars[i] = initial_mean, initial_covar
            else:
                # predict
                prior_means[i], prior_covars[i] = predict(post_means[i-1], post_covars[i-1])

            # update
            post_means[i], post_covars[i], kalman_gains[i], residuals[i], inv_residual_covars[i], covar_updts[i] = \
                update(prior_means[i], prior_covars[i], obs[i])

        mbf_dict = {"residuals": residuals, "inv_residual_covars": inv_residual_covars, "kalman_gains": kalman_gains,
                    "covar_updts": covar_updts}

        return post_means, post_covars, prior_means, prior_covars, mbf_dict

    def backward_pass_rts(self, post_means, post_covars, prior_means, prior_covars):
        smoothed_means = np.zeros(post_means.shape)
        smoothed_covars = np.zeros(post_covars.shape)
        smoothed_means[-1] = post_means[-1]
        smoothed_covars[-1] = post_covars[-1]

        for i in range(len(smoothed_means) - 2, -1, -1):
            c = post_covars[i] @ self._transition_matrix.T @ np.linalg.inv(prior_covars[i+1])
            smoothed_means[i] = post_means[i] + c @ (smoothed_means[i + 1] - prior_means[i + 1])
            smoothed_covars[i] = post_covars[i] + c @ (smoothed_covars[i + 1] - prior_covars[i + 1]) @ c.T
        return smoothed_means, smoothed_covars

    def backward_pass_mfb(self, post_means, post_covars, prior_means, prior_covars, mbf_dict, post=True):
        residuals, inv_residual_covars = mbf_dict["residuals"], mbf_dict["inv_residual_covars"]
        covar_updts = mbf_dict["covar_updts"]

        smoothed_means = np.zeros(post_means.shape)
        smoothed_covars = np.zeros(post_covars.shape)

        lambda_mat = np.zeros([self._state_dim, self._state_dim])
        lambda_vec = np.zeros([self._state_dim, 1])
        if post:
            smoothed_means[-1] = post_means[-1]
            smoothed_covars[-1] = post_covars[-1]

        for i in range(len(smoothed_means) - (2 if post else 1), -1, -1):
            idx = i + 1 if post else i

            lambda_mat = covar_updts[idx].T @ lambda_mat @ covar_updts[idx]
            lambda_mat += self._observation_matrix.T @ inv_residual_covars[idx] @ self._observation_matrix
            if post:
                lambda_mat = self._transition_matrix.T @ lambda_mat @ self._transition_matrix

            lambda_vec = covar_updts[idx].T @ lambda_vec
            lambda_vec += - self._observation_matrix.T @ inv_residual_covars[idx] @ residuals[idx]
            if post:
                lambda_vec = self._transition_matrix.T @ lambda_vec

            if post:
                smoothed_means[i] = post_means[i] - post_covars[i] @ lambda_vec
                smoothed_covars[i] = post_covars[i] - post_covars[i] @ lambda_mat @ post_covars[i]
            else:
                smoothed_means[i] = prior_means[i] - prior_covars[i] @ lambda_vec
                smoothed_covars[i] = prior_covars[i] - prior_covars[i] @ lambda_mat @ prior_covars[i]
        return smoothed_means, smoothed_covars
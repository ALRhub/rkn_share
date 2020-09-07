import tensorflow as tf
from tests.regression_tests.reference_implementation.TF_RKNCell import RKNCellBandMat
from util.ConfigDict import ConfigDict

k = tf.keras


class TFRKNLayer:

    @staticmethod
    def get_default_config():
        config = ConfigDict(
            num_basis=15,
            bandwidth=1,
            time_cont_trans_model=True,
            trans_net_hidden_units=[64, 64],
            trans_net_hidden_activation=tf.nn.tanh,
            learn_trans_covar=True,
            trans_covar=0.1,
            state_dep_trans_covar=True,
            learn_initial_state_covar=False,
            initial_state_covar=1,
            never_invalid=True
        )
        config.finalize_adding()
        return config

    def __init__(self, latent_obs_dim, config):

        self._config = config

        self._lod = latent_obs_dim
        self._lsd = 2 * self._lod

        cell = RKNCellBandMat(obs_dim=self._lod, state_dim=self._lsd, config=config,
                              out_mode=RKNCellBandMat.OUT_MODE_RTS)
        self._layer = k.layers.RNN(cell, return_sequences=True, time_major=False)

    #@tf.function
    def __call__(self, obs_mean, obs_var, obs_valid=None):
        batch_size, seq_length = obs_mean.shape[0], obs_mean.shape[1]
        if obs_valid is None:
            assert self._config.never_invalid
            obs_valid = tf.ones([batch_size, seq_length, 1])


        res = self._layer(tf.concat([obs_mean, obs_var, obs_valid], -1))
        return RKNCellBandMat.unpack_out(res, RKNCellBandMat.OUT_MODE_RTS)

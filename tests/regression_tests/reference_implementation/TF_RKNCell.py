import tensorflow as tf
from tensorflow import keras as k
import numpy as np
from tests.regression_tests.reference_implementation.NetworkBuilder import build_dense_network, NetworkKeys
from typing import List, Tuple
from util import ConfigDict

"""Implementation of the rkn_cell Transition cell, described in 
'Recurrent Kalman Networks:Factorized Inference in High-Dimensional Deep Feature Spaces'
#Todo: add link to paper 
Published at ICML 2019 
Correspondence to: Philipp Becker (philippbecker93@googlemail.com)
"""

TensorList = List[tf.Tensor]


# Math Util
def elup1(x: tf.Tensor) -> tf.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return tf.nn.elu(x) + 1


def elup1_inv(x):
    return np.log(x) if x < 1.0 else (x - 1.0)


def dadat(a: tf.Tensor, diag_mat: tf.Tensor) -> tf.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * A^T) where A is a batch of square matrices and
    diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch of square matrices,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    #diag_ext = tf.expand_dims(diag_mat, 1)
    #first_prod = tf.square(a) * diag_ext
    #ref = tf.reduce_sum(first_prod, axis=2)
    res = tf.linalg.matvec(tf.square(a), diag_mat)
    #tf.print(tf.reduce_max(tf.abs(res - ref)))
    return res


def dadbt(a: tf.Tensor, diag_mat: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * B^T) where A and B are batches of square matrices and
     diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch square matrices
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :param b: batch of square matrices
    :returns diagonal entries of  A * diag_mat * B^T"""
    #diag_ext = tf.expand_dims(diag_mat, 1)
    #first_prod = a * b * diag_ext
    #ref = tf.reduce_sum(first_prod, axis=2)
    res = tf.linalg.matvec(a * b, diag_mat)
    #tf.print(tf.reduce_max(tf.abs(res - ref)))
    return res



class RKNCell(k.layers.Layer):

    OUT_MODE_FILTER = "rkn_out_filter"
    OUT_MODE_RTS = "rkn_out_rts"
    OUT_MODE_DEBUG = "rkn_out_debug"

    """Implementation of the actual transition cell. This is implemented as a subclass of the Keras Layer Class, such
     that it can be used with tf.keras.layers.RNN"""

    # Pack and Unpack functions

    @staticmethod
    def pack_state(mean: tf.Tensor, covar: TensorList) -> tf.Tensor:
        """ packs system state (either prior or posterior) into single vector
        :param mean: state mean as vector
        :param covar: state covar as list [upper, lower, side]
        :return: state as single vector of size 5 * observation dim,
        order of entries: mean, covar_upper, covar_lower, covar_side
        """
        return tf.concat([mean] + covar, -1)

    @staticmethod
    def unpack_state(state: tf.Tensor) -> Tuple[tf.Tensor, TensorList]:
        """ unpacks system state packed by 'pack_state', can be used to unpack cell output (in non-debug case)
        :param state: packed state, containing mean and covar as single vector
        :return: unpacked state
        """
        lod = int(state.get_shape().as_list()[-1] / 5)
        mean = state[..., :2 * lod]
        covar_upper = state[..., 2 * lod: 3 * lod]
        covar_lower = state[..., 3 * lod: 4 * lod]
        covar_side = state[..., 4 * lod:]
        return mean, [covar_upper, covar_lower, covar_side]

    @staticmethod
    def pack_input(obs_mean: tf.Tensor, obs_covar: tf.Tensor, obs_valid: tf.Tensor) -> tf.Tensor:
        """ packs cell input. All inputs provided to the cell should be packed using this function
        :param obs_mean: observation mean
        :param obs_covar: observation covariance
        :param obs_valid: flag indication if observation is valid
        :return: packed input
        """
        if not obs_valid.dtype == tf.float32:
            obs_valid = tf.cast(obs_valid, tf.float32)
        return tf.concat([obs_mean, obs_covar, obs_valid], axis=-1)

    @staticmethod
    def unpack_input(input_as_vector: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """ used to unpack input vectors that where packed with 'pack_input
        :param input_as_vector packed input
        :return: observation mean, observation covar, observation valid flag
        """
        lod = int((input_as_vector.get_shape().as_list()[-1] - 1) / 2)
        obs_mean = input_as_vector[..., :lod]
        obs_covar = input_as_vector[..., lod: -1]
        obs_valid = tf.cast(input_as_vector[..., -1], tf.bool)
        return obs_mean, obs_covar, obs_valid

    @staticmethod
    def pack_debug_output(post_mean: tf.Tensor, post_covar: TensorList,
                          prior_mean: tf.Tensor, prior_covar: TensorList, kalman_gain: TensorList) -> tf.Tensor:
        """
        packs debug output containing...
        :param post_mean: (vector)
        :param post_covar: (list of 3 vectors, covar_upper, covar_lower, covar_side)
        :param prior_mean: (vector)
        :param prior_covar: (list of 3 vectors, covar_upper, covar_lower, covar_side)
        :param kalman_gain: (list of 2 vectors, q_upper, q_lower)
        :return: packed ouptut
        """
        return tf.concat([post_mean] + post_covar + [prior_mean] + prior_covar + kalman_gain, axis=-1)

    @staticmethod
    def unpack_debug_output(output: tf.Tensor) -> Tuple[tf.Tensor, TensorList, tf.Tensor, TensorList, TensorList]:
        """
        :param output: output produced by the cell in debug mode
        :return: unpacked ouptut, i.e.:
                    post_mean: (vector)
                    post_covar: (list of 3 vectors, covar_upper, covar_lower, covar_side)
                    prior_mean: (vector)
                    prior_covar: (list of 3 vectors, covar_upper, covar_lower, covar_side)
                    kalman_gain: (list of 2 vectors, q_upper, q_lower)
        """
        lod = int(output.get_shape().as_list()[-1] / 12)
        post_mean = output[..., :  2 * lod]
        post_covar_upper = output[..., 2 * lod:  3 * lod]
        post_covar_lower = output[..., 3 * lod:  4 * lod]
        post_covar_side = output[..., 4 * lod:  5 * lod]
        prior_mean = output[..., 5 * lod:  7 * lod]
        prior_covar_upper = output[..., 7 * lod:  8 * lod]
        prior_covar_lower = output[..., 8 * lod:  9 * lod]
        prior_covar_side = output[..., 9 * lod: 10 * lod]
        q_upper = output[..., 10 * lod: 11 * lod]
        q_lower = output[..., 11 * lod:]
        post_covar = [post_covar_upper, post_covar_lower, post_covar_side]
        prior_covar = [prior_covar_upper, prior_covar_lower, prior_covar_side]
        return post_mean, post_covar, prior_mean, prior_covar, [q_upper, q_lower]

    @staticmethod
    def unpack_out(output: tf.Tensor, out_mode: str):
        if out_mode == RKNCell.OUT_MODE_FILTER:
            return RKNCell.unpack_state(output)
        elif out_mode == RKNCell.OUT_MODE_RTS:
            lsd = int(output.get_shape().as_list()[-1] / 2)
            post, prior = RKNCell.unpack_state(output[..., :lsd]), RKNCell.unpack_state(output[..., lsd:])
            return post[0], post[1], prior[0], prior[1]
        elif out_mode == RKNCell.OUT_MODE_DEBUG:
            return RKNCell.unpack_debug_output(output)
        else:
            raise AssertionError("Invalid out type")

    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 config: ConfigDict,
                 out_mode: str = OUT_MODE_FILTER):

        """
        :param state_dim: dimensionality of state (n in paper)
        :param obs_dim: dimensionality of observation (m in paper)
        :param number_of_basis: number of basis matrices used (k in paper)
        :param bandwidth: bandwidth of transition sub matrices (b in paper)
        :param trans_net_hidden_units: list of number (numbers of hidden units per layer in coefficient network)
        :param initial_trans_covar: value (scalar) used to initialize transition covariance with
        :param never_invalid: if you know a-priori that the observation valid flag will always be positive you can set
                              this to true for slightly increased performance (obs_valid mask will be ignored)
        :param out_mode: if set the cell output will additionally contain the prior state estimate and kalman gain for
                      debugging/visualization purposes, use 'unpack_debug_output' to unpack.
        """

        super().__init__()

        assert state_dim == 2 * obs_dim, "Currently only 2 * m = n supported"
        self._sd = state_dim
        self._od = obs_dim
        self.c = config
        self._out_mode = out_mode

    def build(self, input_shape: tf.TensorShape):
        """
        see super
        """
        self._build_transition_model()
        self._build_transition_noise_model()

        if self.c.learn_initial_state_covar:
            lic_init = k.initializers.Constant(elup1_inv(self.c.initial_state_covar))
            self._log_init_covar = self.add_weight(shape=[1, self._sd], name="log_initial_covar",
                                                   initializer=lic_init)
        else:
            self._log_init_covar = tf.ones([1, self._sd], dtype=tf.float32) * elup1_inv(self.c.initial_state_covar)

        super().build(input_shape)

    def _build_transition_noise_model(self):

        if self.c.learn_trans_covar:
            tn_init = k.initializers.Constant(elup1_inv(self.c.trans_covar))
            if self.c.state_dep_trans_covar:
                self._trans_noise_raw = self.add_weight(shape=[self.c.num_basis, self._sd], initializer=tn_init,
                                                        name="log_transition_covar_basis")
            else:
                self._trans_noise_raw = self.add_weight(shape=[self._sd], name="log_transition_covar",
                                                        initializer=tn_init)
        else:
            self._trans_noise_raw = tf.ones([self._sd], dtype=tf.float32) * elup1_inv(self.c.trans_covar)

    def _build_transition_model(self):
        raise NotImplementedError

    #@tf.function
    def call(self, inputs: tf.Tensor, states: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Performs one transition step (prediction followed by update in Kalman Filter terms)
        Parameter names match those of superclass - same signature as k.layers.LSTMCell
        :param inputs: Observations (mean and variance vectors concatenated)
        :param states: Last Posterior State (mean and covariance vectors concatenated)
        :return: cell output: current posterior (if not debug, else current posterior, prior and kalman gain)
                 cell state: current posterior
        """
        prior_mean, prior_covar = self.unpack_state(states[0])
        obs_mean, obs_var, obs_valid = self.unpack_input(inputs)

        # update step (current posterior from current prior)

        if self.c.never_invalid:
            update_res = self.update(prior_mean, prior_covar, obs_mean, obs_var)
        else:
            update_res = self.masked_update(prior_mean, prior_covar, obs_mean, obs_var, obs_valid)

        post_mean, post_covar = update_res[:2]

        # predict step (next prior from current posterior (i.e. cell state))

        pred_res = self.predict(post_mean, post_covar)
        prior_mean, prior_covar = pred_res[:2]

        # pack outputs

        prior_state = self.pack_state(prior_mean, prior_covar)
        post_state = self.pack_state(post_mean, post_covar)
        if self._out_mode == RKNCell.OUT_MODE_DEBUG:
            raise NotImplementedError
        elif self._out_mode == RKNCell.OUT_MODE_RTS:
            rts_out = tf.concat([post_mean] + post_covar + [prior_mean] + prior_covar, axis=-1)
            return rts_out, prior_state
        else:
            return post_state, prior_state

    def get_transition_model(self, post_mean: tf.Tensor) -> TensorList:
        raise NotImplementedError

    def predict(self, post_mean: tf.Tensor, post_covar: TensorList) -> Tuple[tf.Tensor, TensorList]:
        raise NotImplementedError

    def masked_update(self, prior_mean, prior_covar, obs_mean, obs_var, obs_valid):
        """ Ensures update only happens if observation is valid
        CAVEAT: You need to ensure that obs_mean and obs_covar do not contain NaNs, even if they are invalid.
        If they do this will cause problems with gradient computation (they will also be NaN) due to how tf.where works
        internally (see: https://github.com/tensorflow/tensorflow/issues/2540)
        :param prior_mean: current prior state mean
        :param prior_covar: current prior state covariance
        :param obs_mean: current observation mean
        :param obs_var: current observation covariance
        :param obs_valid: indicating if observation is valid
        :return: current posterior state mean and covariance
        """
        obs_valid = tf.expand_dims(obs_valid, -1)
        up_res = self.update(prior_mean, prior_covar, obs_mean, obs_var)
        val_mean, val_covar = up_res[:2]
        masked_mean = tf.where(obs_valid, val_mean, prior_mean)

        masked_covar_upper = tf.where(obs_valid, val_covar[0], prior_covar[0])
        masked_covar_lower = tf.where(obs_valid, val_covar[1], prior_covar[1])
        masked_covar_side  = tf.where(obs_valid, val_covar[2], prior_covar[2])

        if self._out_mode == RKNCell.OUT_MODE_DEBUG:
            masked_q_upper = tf.where(obs_valid, up_res[-1][0], tf.zeros(tf.shape(obs_mean)))
            masked_q_lower = tf.where(obs_valid, up_res[-1][1], tf.zeros(tf.shape(obs_mean)))
            return masked_mean, [masked_covar_upper, masked_covar_lower, masked_covar_side], [masked_q_upper, masked_q_lower]
        else:
            return masked_mean, [masked_covar_upper, masked_covar_lower, masked_covar_side]

    def update(self, prior_mean: tf.Tensor, prior_covar: TensorList, obs_mean: tf.Tensor, obs_var: tf.Tensor):
        """Performs update step
        :param prior_mean: current prior state mean
        :param prior_covar: current prior state covariance
        :param obs_mean: current observation mean
        :param obs_var: current covariance mean
        :return: current posterior state and covariance
        """
        covar_upper, covar_lower, covar_side = prior_covar

        # compute kalman gain (eq 2 and 3 in paper)
        denominator = covar_upper + obs_var
        q_upper = covar_upper / denominator
        q_lower = covar_side / denominator

        # update mean (eq 4 in paper)
        residual = obs_mean - prior_mean[:, :self._od]
        new_mean = prior_mean + tf.concat([q_upper * residual, q_lower * residual], -1)

        # update covariance (eq 5 -7 in paper)
        covar_factor = 1 - q_upper
        new_covar_upper = covar_factor * covar_upper
        new_covar_lower = covar_lower - q_lower * covar_side
        new_covar_side = covar_factor * covar_side
        if self._out_mode == RKNCell.OUT_MODE_DEBUG:
            return new_mean, [new_covar_upper, new_covar_lower, new_covar_side], [q_upper, q_lower]
        else:
            return new_mean, [new_covar_upper, new_covar_lower, new_covar_side]

    def get_initial_state(self, inputs: tf.Tensor, batch_size: int, dtype: tf.DType) -> tf.Tensor:
        """
        Signature matches the run required by k.layers.RNN
        :param inputs:
        :param batch_size:
        :param dtype:
        :return:
        """
        init_mean = tf.zeros([batch_size, self._sd])
        init_covar_ul = elup1(self._log_init_covar)
        init_covar_upper = tf.tile(init_covar_ul[:, :self._od], [batch_size, 1])
        init_covar_lower = tf.tile(init_covar_ul[:, self._od:], [batch_size, 1])
        init_covar_side = tf.zeros([batch_size, self._od])
        init_covar = [init_covar_upper, init_covar_lower, init_covar_side]
        return self.pack_state(init_mean, init_covar)

    @property
    def state_size(self):
        """ required by k.layers.RNN"""
        return 5 * self._od

    def compute_output_shape(self, input_shape):
        return 5 * self._od


class RKNCellFullMat(RKNCell):

    def predict(self, post_mean: tf.Tensor, post_covar: TensorList) -> Tuple[tf.Tensor, TensorList]:
        """ Performs prediction step
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :return: current prior state mean and covariance
        """
        # compute state dependent transition matrix
        [tm11, tm12, tm21, tm22], trans_covar = self.get_transition_model(post_mean)

        # prepare transition noise
        trans_covar_upper = trans_covar[..., :self._od]
        trans_covar_lower = trans_covar[..., self._od:]

        # predict next prior mean
        mu = post_mean[:, :self._od]
        ml = post_mean[:, self._od:]

        nmu = tf.linalg.matvec(tm11, mu) + tf.linalg.matvec(tm12, ml)
        nml = tf.linalg.matvec(tm21, mu) + tf.linalg.matvec(tm22, ml)

        # predict next prior covariance (eq 10 - 12 in paper supplement)
        cu, cl, cs = post_covar
        ncu = dadat(tm11, cu) + 2.0 * dadbt(tm11, cs, tm12) + dadat(tm12, cl) + trans_covar_upper
        ncl = dadat(tm21, cu) + 2.0 * dadbt(tm21, cs, tm22) + dadat(tm22, cl) + trans_covar_lower
        ncs = dadbt(tm21, cu, tm11) + dadbt(tm22, cs, tm11) + dadbt(tm21, cs, tm12) + dadbt(tm22, cl, tm12)
        return tf.concat([nmu, nml], axis=-1), [ncu, ncl, ncs]


class RKNCellBandMat(RKNCellFullMat):

    def _build_transition_model(self):
        # build state independent basis matrices
        tile_shape = [self.c.num_basis, 1, 1]
        if self.c.time_cont_trans_model:
            tm_11_init = np.tile(np.expand_dims(np.zeros([self._od, self._od], dtype=np.float32), 0), tile_shape)
            tm_22_init = np.tile(np.expand_dims(np.zeros([self._od, self._od], dtype=np.float32), 0), tile_shape)
        else:
            tm_11_init = np.tile(np.expand_dims(np.eye(self._od, dtype=np.float32), 0), tile_shape) # np.tile(np.expand_dims(np.zeros([self._od, self._od], dtype=np.float32), 0), tile_shape)
            tm_22_init = np.tile(np.expand_dims(np.eye(self._od, dtype=np.float32), 0), tile_shape)
        #tm_11_init = np.tile(np.expand_dims(np.zeros([self._od, self._od], dtype=np.float32), 0), tile_shape)
        tm_12_init =   0.2 * np.tile(np.expand_dims(np.eye(self._od, dtype=np.float32), 0), tile_shape)
        tm_21_init = - 0.2 * np.tile(np.expand_dims(np.eye(self._od, dtype=np.float32), 0), tile_shape)
        #tm_22_init = np.tile(np.expand_dims(np.zeros([self._od, self._od], dtype=np.float32), 0), tile_shape)

        tm_11_full = self.add_weight(shape=[self.c.num_basis, self._od, self._od], name="tm_11_basis",
                                     initializer=k.initializers.Constant(tm_11_init))
        tm_12_full = self.add_weight(shape=[self.c.num_basis, self._od, self._od], name="tm_12_basis",
                                     initializer=k.initializers.Constant(tm_12_init))
        tm_21_full = self.add_weight(shape=[self.c.num_basis, self._od, self._od], name="tm_21_basis",
                                     initializer=k.initializers.Constant(tm_21_init))
        tm_22_full = self.add_weight(shape=[self.c.num_basis, self._od, self._od], name="tm_22_basis",
                                     initializer=k.initializers.Constant(tm_22_init))
        self._transition_matrices_raw = [tm_11_full, tm_12_full, tm_21_full, tm_22_full]

        # build state dependent coefficent network
        c_dict = {NetworkKeys.NUM_UNITS: self.c.trans_net_hidden_units,
                  NetworkKeys.ACTIVATION: self.c.trans_net_hidden_activation}
        self._coefficient_net = build_dense_network(input_dim=self._sd, output_dim=self.c.num_basis,
                                                    output_activation=k.activations.softmax,
                                                    params=c_dict)
        self._coefficient_net.build(input_shape=[None, self._sd])
        self._trainable_weights += self._coefficient_net.weights

    def get_transition_model(self, post_mean: tf.Tensor) -> TensorList:
        # prepare transition model
        coefficients = tf.reshape(self._coefficient_net(post_mean), [-1, self.c.num_basis, 1, 1])

        tm11 = tf.reduce_sum(tf.expand_dims(self._transition_matrices_raw[0], 0) * coefficients, 1)
        tm11 = tf.linalg.band_part(tm11, self.c.bandwidth, self.c.bandwidth)

        tm12 = tf.linalg.band_part(self._transition_matrices_raw[1], self.c.bandwidth, self.c.bandwidth)
        tm12 = tf.reduce_sum(tf.expand_dims(tm12, 0) * coefficients, 1)

        tm21 = tf.linalg.band_part(self._transition_matrices_raw[2], self.c.bandwidth, self.c.bandwidth)
        tm21 = tf.reduce_sum(tf.expand_dims(tm21, 0) * coefficients, 1)

        tm22 = tf.linalg.band_part(self._transition_matrices_raw[3], self.c.bandwidth, self.c.bandwidth)
        tm22 = tf.reduce_sum(tf.expand_dims(tm22, 0) * coefficients, 1)

        if self.c.time_cont_trans_model:
            tm11 += tf.eye(self._od, batch_shape=[1])
            tm22 += tf.eye(self._od, batch_shape=[1])

        if self.c.state_dep_trans_covar:
            log_trans_std = tf.reduce_sum(tf.expand_dims(self._trans_noise_raw, 0) * coefficients[..., 0], 1)
            trans_cov = elup1(log_trans_std)
        else:
            log_trans_cov = tf.expand_dims(self._trans_noise_raw, 0)
            trans_cov = elup1(log_trans_cov)

        return [tm11, tm12, tm21, tm22], trans_cov
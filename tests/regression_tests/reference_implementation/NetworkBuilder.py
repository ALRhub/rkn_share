import tensorflow as tf
import numpy as np

k = tf.keras

class NetworkKeys:
    NUM_UNITS = "num_units"
    ACTIVATION = "activation"
    L2_REG_FACT = "l2_reg_fact"
    DROP_PROB = "drop_prob"
    BATCH_NORM = "batch_norm"
    OUTPUT_INITIALIZER = "output_initializer"

class GlorotFactInitializer(k.initializers.Initializer):

    def __init__(self, fact=1.0):
        self._fact = fact

    def __call__(self, shape, dtype=None):
        dtype = dtype if dtype is not None else tf.float32
        limit = self._fact * np.sqrt(6 / (shape[-1] + shape[-2]))
        return tf.random.uniform(minval=-limit, maxval=limit, shape=shape, dtype=dtype)

    def from_config(cls, config):
        return cls(config["factor"])

    def get_config(self):
        return {"factor": self._fact}



def build_dense_network(input_dim: int, output_dim: int, output_activation,
                        params: dict, with_output_layer: bool = True) -> k.models.Sequential:
    """Builds a simple feed forward network"""

    model = k.models.Sequential()

    activation = params.get(NetworkKeys.ACTIVATION, "relu")
    l2_reg_fact = params.get(NetworkKeys.L2_REG_FACT, 0.0)
    regularizer = k.regularizers.l2(l2_reg_fact) if l2_reg_fact > 0 else None
    drop_prob = params.get(NetworkKeys.DROP_PROB, 0.0)
    batch_norm = params.get(NetworkKeys.BATCH_NORM, False)
    output_initializer = params.get(NetworkKeys.OUTPUT_INITIALIZER, "glorot_uniform")

    last_dim = input_dim
    for i in range(len(params[NetworkKeys.NUM_UNITS])):
        model.add(k.layers.Dense(units=params[NetworkKeys.NUM_UNITS][i],
                                 kernel_regularizer=regularizer,
                                 input_dim=last_dim))
        if batch_norm:
            model.add(k.layers.BatchNormalization())
        model.add(k.layers.Activation(activation))
        last_dim = params[NetworkKeys.NUM_UNITS][i]

        if drop_prob > 0.0:
            model.add(k.layers.Dropout(rate=drop_prob))
    if with_output_layer:
        model.add(k.layers.Dense(units=output_dim,
                                 activation=output_activation,
                                 kernel_initializer=output_initializer))
    return model

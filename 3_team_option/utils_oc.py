import numpy as np
import tensorflow as tf
from utils_transformer import make_mask, make_padded_obs


def get_option(agents_q_omega, mask, policy_over_options):
    """
    batch_size: b=1
    :param agents_q_omega: (b,n,option_dim)
    :param mask: (b,n)
    :return: option, ndarray (1,)
    """

    broadcast_float_mask = tf.expand_dims(tf.cast(mask, 'float32'), axis=-1)  # (b,n,1)
    agents_q_omega = agents_q_omega * broadcast_float_mask  # (b,n,option_dim)
    Q_Omega = tf.reduce_sum(agents_q_omega, axis=1)  # (b,option_dim)

    option = policy_over_options.sample(Q_Omega)  # int
    option = np.array([option])  # (1,)

    return option  # ndarray, (1,)

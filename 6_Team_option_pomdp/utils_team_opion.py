"""
Copied from /4_Applying_Option_Critic_to_MARL_Transformer/3_team_option/utils_oc.py
"""
import numpy as np
import tensorflow as tf


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
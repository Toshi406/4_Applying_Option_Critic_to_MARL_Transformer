import numpy as np
import tensorflow as tf
from utils_transformer_mtc_dec_pomdp import make_mask, make_padded_obs


def get_options(agents_q_omega, mask, policy_over_options, alive_agents_ids):
    """
    batch_size: b=1
    :param agents_q_omega: (b,n,option_dim)
    :param mask: (b,n)
    alive_agents_ids: list of int
    :return: option, ndarray (1,n), option=option_dim+1 for dead/dummy agents
    """

    broadcast_float_mask = tf.expand_dims(tf.cast(mask, 'float32'), axis=-1)  # (b,n,1)
    agents_q_omega = agents_q_omega * broadcast_float_mask  # (b,n,option_dim)

    option = policy_over_options.sample(agents_q_omega, alive_agents_ids)  # (1,n), ndarray

    return option  # ndarray, (1,n)


def get_agent_option(agent_q_omega, policy_over_options):
    # qgent_q_omega: Q_Ω^i, (1,option_dim)
    option = policy_over_options.sample_agent_option(agent_q_omega)
    return option  # int

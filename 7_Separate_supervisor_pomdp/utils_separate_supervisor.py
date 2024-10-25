"""
Copied from /6_team_option_pomdp/utils_team_option.py, then modified.
"""
import numpy as np
import tensorflow as tf


def get_option(Q_Omega, policy_over_options):
    """
    batch_size: b=1
    :param Q_Omega (Q_Ω(s,ω)): (b,option_dim)
    :return: option, ndarray (1,)
    """

    option = policy_over_options.sample(Q_Omega)  # int
    option = np.array([option])  # (1,)

    return option  # ndarray, (1,)
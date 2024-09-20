import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from config_oc_rev import Config
from sub_models_indiv_option import CNNModel, MultiHeadAttentionModel, AgentPolicyHead, \
    AgentQHead
from global_models_indv_option import GlobalCNNModel, TerminationHead
from utils_transformer import make_mask, make_padded_obs


class EpsGreedyOverOptions:
    def __init__(self, config):
        self.noptions = config.option_dim
        self.epsilon = config.epsilon

    def sample(self, Q_Omega):
        """
        :param Q_Omega #(1,noptions) :: Note: batch should be 1
        :return: omega  # int
        """

        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.noptions)  # int
        else:
            q_omega = Q_Omega.numpy()  # (1,noptions)
            omega = np.argmax(q_omega, axis=-1)  # (1,)
            return omega[0]  # int


class AgentFeatureModel(tf.keras.models.Model):
    """
    :inputs: padded obs=(None,n,g,g,ch*n_frames),
             mask (None,n,n),  n=max_num_agents
    :return: agent_features: (None,n,hidden_dim)
             [score1, score2]: [(None,num_heads,n,n),(None,num_heads,n,n)]

    Model: "agent_feature_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     cnn_model (CNNModel)        multiple                  258944

     time_distributed_6 (TimeDis  multiple                 0
     tributed)

     multi_head_attention_model   multiple                 526080
     (MultiHeadAttentionModel)

     multi_head_attention_model_  multiple                 526080
     1 (MultiHeadAttentionModel)

    =================================================================
    Total params: 1,311,104
    Trainable params: 1,311,104
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(AgentFeatureModel, self).__init__(**kwargs)

        self.config = config

        self.cnn = CNNModel(config=self.config)

        self.dropout = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        )

        self.mha1 = MultiHeadAttentionModel(config=self.config)

        self.mha2 = MultiHeadAttentionModel(config=self.config)

    @tf.function
    def call(self, x, mask, training=False):
        """
        x=agents_obs=(None,n,g,g,ch*n_frames)=(None,15,15,15,6*1),
        mask:(None,n)=(None,15)
        """

        """ CNN layer """
        features_cnn = self.cnn(x, mask)  # (None,n,hidden_dim)

        """ Dropout layer """
        features_cnn = self.dropout(features_cnn, training=training)

        """ Multi Head Self-Attention layer 1 """
        # features_mha1: (None,n,hidden_dim),
        # score1: (None,num_heads,n,n)
        features_mha1, score1 = self.mha1(features_cnn, mask, training=training)

        """ Multi Head Self-Attention layer 2 """
        # features_mha2: (None,n,hidden_dim),
        # score2: (None,num_heads,n,n)
        features_mha2, score2 = self.mha2(features_mha1, mask, training=training)

        return features_mha2, [score1, score2]

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames)
        )

        features_cnn = self.cnn(x, mask)

        features_cnn = self.dropout(features_cnn, training=True)

        features_mha1, score1 = self.mha1(features_cnn, mask, training=True)
        features_mha2, score2 = self.mha2(features_mha1, mask, training=True)

        model = tf.keras.models.Model(
            inputs=x,
            outputs=[features_mha2, [score1, score2]],
            name='agent_features',
        )

        return model


class Individual_OptionCritic_Model(tf.keras.models.Model):
    """
    Model: "individual__option_critic__model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     global_cnn_model_1 (GlobalC  multiple                 258944
     NNModel)

     agent_feature_model_1 (Agen  multiple                 1311104
     tFeatureModel)

     termination_head (Terminati  multiple                 16708
     onHead)

     agent_q_head_1 (AgentQHead)  multiple                 591876

     agent_policy_head (AgentPol  multiple                 399380
     icyHead)

    =================================================================
    Total params: 2,578,012
    Trainable params: 2,578,012
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(Individual_OptionCritic_Model, self).__init__(**kwargs)

        self.config = config

        self.global_cnn = GlobalCNNModel(config)
        self.agents_feature_model = AgentFeatureModel(config)
        self.termination_head = TerminationHead(config)
        self.agents_q_head = AgentQHead(config)
        self.agents_policy_head = AgentPolicyHead(config)

    def call(self, inputs, mask):
        """
        :param inputs: [agents_state, global_state],
                     # [(b,n,g,g,ch*nframes),(b,g,g,global_ch*global_n_frames)
        :param mask: (b,n)
        :return:
            termination_probs: (b,option_dim)
            agents_q_omegas: (b,n,option_dim)
            agents_policy_logits: (b,n,option_dim,action_dim)
            scores=[score_1,score_2]: [(b,num_heads,n,n),(b,num_heads,n,n)], num_heads=2
        """

        agents_state = inputs[0]
        global_state = inputs[1]

        global_feature = self.global_cnn(global_state)  # (b,hidden_dim)
        agents_feature, scores = self.agents_feature_model(agents_state, mask, training=False)
        # (b,n,hidden_dim), [(b,num_heads,n,n),(b,num_heads,n,n)], num_heads=2

        termination_probs = self.termination_head(global_feature)  # (b,option_dim)

        agents_q_omega = self.agents_q_head([agents_feature, global_feature], mask, training=False)
        # (b,n,option_dim)

        agents_policy_logit = self.agents_policy_head(agents_feature, mask, training=False)
        # (b,n,option_dim,action_dim)

        return termination_probs, agents_q_omega, agents_policy_logit, scores


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    grid_size = config.grid_size

    """ global_state & features """
    global_ch = config.global_observation_channels  # 6
    global_n_frames = config.global_n_frames

    global_state_shape = (grid_size, grid_size, global_ch * global_n_frames)  # (15,15,6)

    global_state = np.ones(shape=global_state_shape)  # (15,15,6)
    global_state = np.expand_dims(global_state, axis=0)  # (1,15,15,6)

    # global_feature
    global_cnn = GlobalCNNModel(config=config)
    global_feature = global_cnn(global_state)  # (1,256)

    """ agents obs """
    ch = config.observation_channels
    n_frames = config.n_frames

    obs_shape = (grid_size, grid_size, ch * n_frames)

    max_num_agents = config.max_num_red_agents

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]
    agent_obs = {}

    for i in alive_agents_ids:
        agent_id = 'red_' + str(i)
        agent_obs[agent_id] = np.ones(obs_shape)

    # Get padded_obs and mask
    padded_obs = make_padded_obs(max_num_agents, obs_shape, agent_obs)  # (1,n,g,g,ch*n_frames)

    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    """ Make model """
    agent_feature = AgentFeatureModel(config=config)

    features, scores = agent_feature(padded_obs, mask, training=True)  # (1,n,hidden_dim)

    agent_feature.summary()

    # Q_Omega
    q_models = AgentQHead(config=config)
    q_omegas = q_models([features, global_feature], mask)  # (b,n,option_dim)

    broadcast_float_mask = tf.expand_dims(tf.cast(mask, 'float32'), axis=-1)  # (b,n,1)
    q_omegas = q_omegas * broadcast_float_mask  # (b,n,1)

    Q_Omega = tf.reduce_sum(q_omegas, axis=1)  # (b,option_dim)

    # policy over options
    policy_over_options = EpsGreedyOverOptions(config=config)
    option = policy_over_options.sample(Q_Omega)  # int
    print(f'Q_Omega: {Q_Omega},    option: {option}')

    """ Option Critic Model """
    oc_model = Individual_OptionCritic_Model(config)

    termination_probs, agents_q_omega, agents_policy_logit, scores = \
        oc_model([padded_obs, global_state], mask)

    policy_probs, log_policy = oc_model.agents_policy_head.policy_pdf(agents_policy_logit, mask)
    # (b,n,option_dim,action_dim), (b,n,option_dim,action_dim)

    options = np.array([option])  # (1,)
    action = oc_model.agents_policy_head.sample_actions(agents_policy_logit, options, mask)
    print(action)  # (b,n), ndarray, 5 for dead/dummy agents

    """ Summary """
    oc_model.summary()


if __name__ == '__main__':
    main()

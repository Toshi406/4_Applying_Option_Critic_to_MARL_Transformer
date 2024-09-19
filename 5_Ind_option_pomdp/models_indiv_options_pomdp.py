import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from config_oc_pomdp import Config
from sub_models_indiv_option_pomdp import CNNModel, MultiHeadAttentionModel, AgentPolicyHead, \
    AgentQHead, AgentTerminationHead
from global_models_indv_option import GlobalCNNModel
from utils_transformer_mtc_dec_pomdp import make_mask, make_padded_obs, make_padded_pos


class EpsGreedyOverOptions:
    def __init__(self, config):
        self.noptions = config.option_dim
        self.epsilon = config.epsilon
        self.max_num_agents = config.max_num_red_agents

    def sample(self, Q_Omega, alive_agents_ids):
        """
        :param Q_Omega #(1,n,option_dim) :: Note: batch should be 1
               alive_agents_ids :: list of int (alive agent_id)
        :return: omega  # (n,), np.array
        """

        omega = np.ones((1, self.max_num_agents)) * (self.noptions + 1)  # (1,n)

        for i in alive_agents_ids:

            if np.random.uniform() < self.epsilon:
                omega[0, i] = np.random.randint(self.noptions)
            else:
                q_omega = Q_Omega.numpy()  # (1,n, option_dim)
                omega[0, i] = np.argmax(q_omega[0, i, :], axis=-1)

        return omega  # (1,n)

    def sample_agent_option(self, agent_q_omega):
        # agent_q_omega: Q_Ω^i, (1,option_dim)

        if np.random.uniform() < self.epsilon:
            omega = np.random.randint(self.noptions)  # int
        else:
            omega = np.argmax(agent_q_omega, axis=-1)[0]  # int

        return omega  # int


class POAgentFeatureModel(tf.keras.models.Model):
    """
    :inputs: [padded_obs, padded_pos]
                padded obs: (None,n,g,g,ch*n_frames), n=max_num_agents
                padded_pos: (None,n,2*n_frames),
             (alive) mask: (None,n), bool
             attention_mask: (None,n,n), bool
    :return: agent_features: (None,n,hidden_dim)
             [score1, score2]: [(None,num_heads,n,n),(None,num_heads,n,n)]

    Model: "po_agent_feature_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     cnn_model (CNNModel)        multiple                  233120

     time_distributed_7 (TimeDis  multiple                 0
     tributed)

     multi_head_attention_model   multiple                 49792
     (MultiHeadAttentionModel)

     multi_head_attention_model_  multiple                 49792
     1 (MultiHeadAttentionModel)

    =================================================================
    Total params: 332,704
    Trainable params: 332,704
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(POAgentFeatureModel, self).__init__(**kwargs)

        self.config = config

        self.cnn = CNNModel(config=self.config)

        self.dropout = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        )

        self.mha1 = MultiHeadAttentionModel(config=self.config)

        self.mha2 = MultiHeadAttentionModel(config=self.config)

    @tf.function
    def call(self, x, mask, attention_mask, training=False):
        """
        x=[agents_obs, agents_pos]
            agents_obs: (None,n,2*fov+1,2*fov+1,ch*n_frames)=(None,15,5,5,4*4),
            agents_pos: (None,n,2*n_frames)=(None,15,2*4)
        (alive) mask: (None,n)=(None,15), bool
        attention_mask: (None,n,n)=(None,15,15), bool
        """

        """ CNN layer """
        features_cnn = self.cnn(x, mask)  # (None,n,hidden_dim)

        """ Dropout layer """
        features_cnn = self.dropout(features_cnn, training=training)

        """ Multi Head Self-Attention layer 1 """
        # features_mha1: (None,n,hidden_dim),
        # score1: (None,num_heads,n,n)
        features_mha1, score1 = self.mha1(features_cnn, mask, attention_mask, training=training)

        """ Multi Head Self-Attention layer 2 """
        # features_mha2: (None,n,hidden_dim),
        # score2: (None,num_heads,n,n)
        features_mha2, score2 = self.mha2(features_mha1, mask, attention_mask, training=training)

        return features_mha2, [score1, score2]

    def build_graph(self, mask, attention_mask):
        x1 = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames)
        )

        x2 = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents, 2 * self.config.n_frames)
        )

        x = [x1, x2]

        features_cnn = self.cnn(x, mask)

        features_cnn = self.dropout(features_cnn, training=True)

        features_mha1, score1 = self.mha1(features_cnn, mask, attention_mask, training=True)
        features_mha2, score2 = self.mha2(features_mha1, mask, attention_mask, training=True)

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
     global_cnn_model_1 (GlobalC  multiple                 148160
     NNModel)

     po_agent_feature_model_1 (P  multiple                 332704
     OAgentFeatureModel)

     agent_termination_head (Age  multiple                 37380
     ntTerminationHead)

     agent_q_head_1 (AgentQHead)  multiple                 37380

     agent_policy_head (AgentPol  multiple                 26132
     icyHead)

    =================================================================
    Total params: 581,756
    Trainable params: 581,756
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(Individual_OptionCritic_Model, self).__init__(**kwargs)

        self.config = config

        self.global_cnn = GlobalCNNModel(config)
        self.agents_feature_model = POAgentFeatureModel(config)
        self.agents_termination_head = AgentTerminationHead(config)
        self.agents_q_head = AgentQHead(config)
        self.agents_policy_head = AgentPolicyHead(config)

    def call(self, inputs, mask, attention_mask):
        """
        :param inputs: [agents_state, global_state],
            agent_state: [agent_obs, agent_pos]=[(b,n,2*fov+1,2*fov+1,ch*n_frames),(b,n,2*n_frames)]
            global_state: (b,global_g,global_g,global_ch*global_n_frames)
        :param (alive) mask: (b,n)
        :param attention_mask: (b,n,n)
        :return:
            termination_probs: (b,option_dim)
            agents_q_omegas: (b,n,option_dim)
            agents_policy_logits: (b,n,option_dim,action_dim)
            scores=[score_1,score_2]: [(b,num_heads,n,n),(b,num_heads,n,n)], num_heads=2
        """

        agents_state = inputs[0]
        global_state = inputs[1]

        global_feature = self.global_cnn(global_state)  # (b,hidden_dim)
        agents_feature, scores = (
            self.agents_feature_model(agents_state, mask, attention_mask, training=False))
        # (b,n,hidden_dim), [(b,num_heads,n,n),(b,num_heads,n,n)], num_heads=2

        agents_termination_probs = \
            self.agents_termination_head([agents_feature, global_feature], mask)
        # (b,n,option_dim)

        agents_q_omega = (
            self.agents_q_head([agents_feature, global_feature], mask, training=False))
        # (b,n,option_dim)

        agents_policy_logit = self.agents_policy_head(agents_feature, mask, training=False)
        # (b,n,option_dim,action_dim)

        return agents_termination_probs, agents_q_omega, agents_policy_logit, scores


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    grid_size = config.grid_size

    fov = config.fov
    com = config.com

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

    obs_shape = (2 * fov + 1, 2 * fov + 1, ch * n_frames)  # (5,5,16)
    pos_shape = (2 * n_frames,)  # (8,)

    max_num_agents = config.max_num_red_agents

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]
    agent_obs = {}
    agent_pos = {}

    for i in alive_agents_ids:
        agent_id = 'red_' + str(i)
        agent_obs[agent_id] = np.ones(obs_shape)
        agent_pos[agent_id] = np.ones(pos_shape) * i

    # Get padded_obs and mask
    padded_obs = make_padded_obs(max_num_agents, obs_shape, agent_obs)
    # (1,n,2*fov+1,2*fov+1,ch*n_frames)=(1,15,5,5,4*4)

    padded_pos = make_padded_pos(max_num_agents, pos_shape, agent_pos)
    # (1,n,2*n_frames)=(1,15,2*4)

    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    # Get attention mask (adjacency matrix)
    float_mask = \
        tf.expand_dims(
            tf.cast(mask, 'float32'),
            axis=-1
        )  # (1,n,1)

    attention_mask = tf.matmul(
        float_mask, float_mask, transpose_b=True
    )  # (1,n,n)

    attention_mask = tf.cast(attention_mask, 'bool')

    """ Make model """
    agent_feature = POAgentFeatureModel(config=config)

    features, scores = (
        agent_feature([padded_obs, padded_pos], mask, attention_mask, training=True))
    # (1,n,hidden_dim), [(1,2,n,n),(1,2,n,n)]

    agent_feature.summary()

    # Q_Omega
    q_models = AgentQHead(config=config)
    q_omegas = q_models([features, global_feature], mask)  # (b,n,option_dim)

    broadcast_float_mask = tf.expand_dims(tf.cast(mask, 'float32'), axis=-1)  # (b,n,1)
    q_omegas = q_omegas * broadcast_float_mask  # (b,n,option_dim)

    # policy over options
    policy_over_options = EpsGreedyOverOptions(config=config)
    options = policy_over_options.sample(q_omegas, alive_agents_ids)  # (1,n)
    print(f'alive_agents_ids: {alive_agents_ids}, options: {options}')

    """ Option Critic Model """
    oc_model = Individual_OptionCritic_Model(config)

    termination_probs, agents_q_omega, agents_policy_logit, scores = \
        oc_model([[padded_obs, padded_pos], global_state], mask, attention_mask)

    policy_probs, log_policy = oc_model.agents_policy_head.policy_pdf(agents_policy_logit, mask)
    # (b,n,option_dim,action_dim), (b,n,option_dim,action_dim)

    action = oc_model.agents_policy_head.sample_actions(agents_policy_logit, options, mask)
    print(action)  # (b,n), ndarray, 5 for dead/dummy agents

    termination_bools = \
        oc_model.agents_termination_head.sample_termination(termination_probs, options)
    # batch should be 1, (1,n), bool, False for dead/dummy agents

    """ Summary """
    oc_model.summary()


if __name__ == '__main__':
    main()

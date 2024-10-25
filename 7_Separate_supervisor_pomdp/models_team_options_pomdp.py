"""
    Copied from "4_Applying_Option_Critic_to_MARL_Transformer/6_Team_option_pomdp/
    models_team_options_pomdp.py". Then modified
"""
import numpy as np
import tensorflow as tf
import os

from keras.utils.version_utils import training

from config_separate_supervisor import Config
from sub_models_indiv_option_pomdp import CNNModel, MultiHeadAttentionModel, AgentPolicyHead
from global_models_team_option import GlobalCNNModel, TerminationHead, QHead
from utils_transformer_mtc_dec_pomdp import make_mask, make_padded_obs, make_padded_pos


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


class POAgentsModel(tf.keras.models.Model):
    """
    :inputs: [padded_obs, padded_pos]
                padded obs: (None,n,g,g,ch*n_frames), n=max_num_agents
                padded_pos: (None,n,2*n_frames),
             (alive) mask: (None,n), bool
             attention_mask: (None,n,n), bool
    :return: agents_policy_logits: (b,n,option_dim,action_dim)
             scores=[score_1,score_2]: [(b,num_heads,n,n),(b,num_heads,n,n)], num_heads=2

    Model: "po_agents_model"
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

     agent_policy_head (AgentPol  multiple                 26132
     icyHead)

    =================================================================
    Total params: 358,836
    Trainable params: 358,836
    Non-trainable params: 0
    _________________________________________________________________

    """

    def __init__(self, config, **kwargs):
        super(POAgentsModel, self).__init__(**kwargs)

        self.config = config

        self.cnn = CNNModel(config=self.config)

        self.dropout = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        )

        self.mha1 = MultiHeadAttentionModel(config=self.config)

        self.mha2 = MultiHeadAttentionModel(config=self.config)

        self.agents_policy_head = AgentPolicyHead(config)

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
        features_mha2, score2 = self.mha2(features_mha1, mask, attention_mask,
                                          training=training)

        agents_policy_logit = self.agents_policy_head(features_mha2, mask, training=False)
        # (b,n,option_dim,action_dim)

        return agents_policy_logit, [score1, score2]


class SupervisorModel(tf.keras.models.Model):
    """

    """

    def __init__(self, config, **kwargs):
        super(SupervisorModel, self).__init__(**kwargs)

        self.config = config

        self.global_cnn = GlobalCNNModel(config)
        self.termination_head = TerminationHead(config)  # modified.
        self.q_head = QHead(config)

    @tf.function
    def call(self, inputs):
        """
        :param inputs: global_state: (b,global_g,global_g,global_ch*global_n_frames)
        :return:
            termination_prob: (b,option_dim)
            Q_Ω(s,ω): (b,option_dim)

        Model: "supervisor_model"
        _________________________________________________________________
         Layer (type)                Output Shape              Param #
        =================================================================
         global_cnn_model (GlobalCNN  multiple                 148160
         Model)

         termination_head (Terminati  multiple                 4420
         onHead)

         q_head (QHead)              multiple                  6372

        =================================================================
        Total params: 158,952
        Trainable params: 158,952
        Non-trainable params: 0
        _________________________________________________________________
        """

        global_feature = self.global_cnn(inputs)  # (b,64)

        termination_prob = self.termination_head(global_feature)  # (b,option_dim)

        q_omega = self.q_head(global_feature, training=False)  # (b,option_dim)

        return termination_prob, q_omega

    def build_graph(self):
        x = tf.keras.layers.Input(
            shape=(self.config.global_grid_size,
                   self.config.global_grid_size,
                   self.config.global_n_frames * self.config.global_observation_channels)
        )

        global_feature = self.global_cnn(x)
        termination_prob = self.termination_head(global_feature)
        q_omega = self.q_head(global_feature, training=False)

        model = tf.keras.models.Model(
            inputs=x,
            outputs=[termination_prob, q_omega],
            name='supervisor',
        )

        return model


def main():
    """
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    """

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

    """ Supervisor model """
    supervisor = SupervisorModel(config)

    termination_probs, Q_Omega = supervisor(global_state)  # (1,option_dim), (1,option_dim)

    supervisor.summary()

    # policy over options
    policy_over_options = EpsGreedyOverOptions(config=config)
    options = policy_over_options.sample(Q_Omega)  # int
    print(f'Q_Omega: {Q_Omega}, options: {options}')

    termination_bools = \
        supervisor.termination_head.sample_termination(termination_probs, options)
    # batch should be 1, (1,1), bool, False for dead/dummy agents

    print(f'termination_bool: {termination_bools[0, 0]}')

    tf.keras.utils.plot_model(
        supervisor.build_graph(),
        to_file='supervisor',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )

    """ Agents model """
    po_agents = POAgentsModel(config=config)

    agents_policy_logit, scores = (
        po_agents([padded_obs, padded_pos], mask, attention_mask, training=True))
    # (1,n,option_dim,action_dim), [(1,2,n,n),(1,2,n,n)]

    po_agents.summary()

    action = po_agents.agents_policy_head.sample_actions(agents_policy_logit, options, mask)
    print(action)  # (b,n), ndarray, 5 for dead/dummy agents


if __name__ == '__main__':
    main()

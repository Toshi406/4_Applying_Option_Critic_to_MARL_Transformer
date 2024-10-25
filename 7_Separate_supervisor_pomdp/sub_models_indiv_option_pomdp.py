"""
    Copied from "4_Applying_Option_Critic_to_MARL_Transformer/6_Team_option_pomdp/
    sub_models_team_option.py". Delete AgentQHead & AgentTerminationHead
"""
import os.path

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from config_separate_supervisor import Config
from utils_transformer_mtc_dec_pomdp import make_mask, make_padded_obs, make_padded_pos
from global_models_team_option import GlobalCNNModel


class CNNModel(tf.keras.models.Model):
    """
    # Add AutoEncoder
    :param max_num_agents=n=15
    :inputs: [obs, pos], obs: (None,n,2*fov+1,2*fov+1,ch*n_frames), pos: (None,n,2*n_frames)
    :return: (None,n,hidden_dim)=(None,15,64)

    Model: "cnn_model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_1 (InputLayer)           [(None, 15, 5, 5, 1  0           []
                                    6)]

     time_distributed (TimeDistribu  (None, 15, 5, 5, 64  1088       ['input_1[0][0]']
     ted)                           )

     time_distributed_1 (TimeDistri  (None, 15, 3, 3, 12  73856      ['time_distributed[0][0]']
     buted)                         8)

     time_distributed_2 (TimeDistri  (None, 15, 1, 1, 12  147584     ['time_distributed_1[0][0]']
     buted)                         8)

     input_2 (InputLayer)           [(None, 15, 8)]      0           []

     time_distributed_3 (TimeDistri  (None, 15, 128)     0           ['time_distributed_2[0][0]']
     buted)

     time_distributed_4 (TimeDistri  (None, 15, 32)      288         ['input_2[0][0]']
     buted)

     time_distributed_5 (TimeDistri  (None, 15, 160)     0           ['time_distributed_3[0][0]',
     buted)                                                           'time_distributed_4[0][0]']

     time_distributed_6 (TimeDistri  (None, 15, 64)      10304       ['time_distributed_5[0][0]']
     buted)

     tf.math.multiply (TFOpLambda)  (None, 15, 64)       0           ['time_distributed_6[0][0]']

    ==================================================================================================
    Total params: 233,120
    Trainable params: 233,120
    Non-trainable params: 0
    __________________________________________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(CNNModel, self).__init__(**kwargs)

        self.config = config

        self.conv0 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=1,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.flatten1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )

        self.dense_pos_enc = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=32,
                                      activation='relu',
                                      )
            )

        self.concatenate = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Concatenate(axis=-1)
            )

        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation=None
                )
            )

    @tf.function
    def call(self, inputs, mask):
        # inputs: [obs, pos],
        #   obs: (None,n,2*fov+1,2*fov+1,ch*n_frames)=(None,15,5,5,16),
        #   pos: (None,n,2*n_frames)=(None,15,8)
        # (alive_)mask: (b,n)=(1,15), bool

        h = self.conv0(inputs[0])  # (1,15,5,5,64)
        h = self.conv1(h)  # (1,15,3,3,128)
        h = self.conv2(h)  # (1,15,1,1,128)

        h1 = self.flatten1(h)  # (1,15,128)

        pos_enc = self.dense_pos_enc(inputs[1])  # (1,15,32)

        z = self.concatenate([h1, pos_enc])  # (1,15,160)

        features = self.dense1(z)  # (1,15,64)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (1,15,1)

        features = features * broadcast_float_mask  # (1,15,64)

        return features

    def build_graph(self, mask):
        """ For summary & plot_model """
        x0 = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   2 * self.config.fov + 1,
                   2 * self.config.fov + 1,
                   self.config.observation_channels * self.config.n_frames)
        )

        x1 = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents, 2 * self.config.n_frames)
        )

        x = [x0, x1]

        model = \
            tf.keras.models.Model(
                inputs=[x],
                outputs=self.call(x, mask),
                name='cnn_model'
            )

        return model


class MultiHeadAttentionModel(tf.keras.models.Model):
    """
    Two layers of MultiHeadAttention (Self Attention with provided mask)

    :param mask: (None,n,n), bool
    :param max_num_agents=15=n
    :param hidden_dim = 64

    :return: features: (None,n,hidden_dim)=(None,15,64)
             score: (None,num_heads,n,n)=(None,2,15,15)

    Model: "multi_head_attention_model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_5 (InputLayer)           [(None, 15, 64)]     0           []

     multi_head_attention (MultiHea  ((None, 15, 64),    33216       ['input_5[0][0]',
     dAttention)                     (None, 2, 15, 15))               'input_5[0][0]',
                                                                      'input_5[0][0]']

     add (Add)                      (None, 15, 64)       0           ['input_5[0][0]',
                                                                      'multi_head_attention[0][0]']

     dense_3 (Dense)                (None, 15, 128)      8320        ['add[0][0]']

     dense_4 (Dense)                (None, 15, 64)       8256        ['dense_3[0][0]']

     dropout (Dropout)              (None, 15, 64)       0           ['dense_4[0][0]']

     add_1 (Add)                    (None, 15, 64)       0           ['add[0][0]',
                                                                      'dropout[0][0]']

     tf.math.multiply_2 (TFOpLambda  (None, 15, 64)      0           ['add_1[0][0]']
     )

    ==================================================================================================
    Total params: 49,792
    Trainable params: 49,792
    Non-trainable params: 0
    __________________________________________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(MultiHeadAttentionModel, self).__init__(**kwargs)

        self.config = config

        self.mha1 = \
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=self.config.key_dim,
            )

        self.add1 = \
            tf.keras.layers.Add()

        """
        self.layernorm1 = \
            tf.keras.layers.LayerNormalization(
                axis=-1, center=True, scale=True
            )
        """

        self.dense1 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim * 2,
                activation='relu',
            )

        self.dense2 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim,
                activation=None,
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.add2 = tf.keras.layers.Add()

        """
        self.layernorm2 = \
            tf.keras.layers.LayerNormalization(
                axis=-1, center=True, scale=True
            )
        """

    @tf.function
    def call(self, inputs, mask=None, attention_mask=None, training=False):
        # inputs: (None,n,hidden_dim)=(None,15,64)
        # mask: alive agent mask, (None,n)=(None,15), bool
        # attention_mask(alive+com mask=adjacency matrix): (None,n,n)=(None,15,15), bool

        x, score = \
            self.mha1(
                query=inputs,
                key=inputs,
                value=inputs,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )  # (None,n,hidden_dim),(None,num_heads,n,n)=(None,15,64),(None,2,15,15)

        x1 = self.add1([inputs, x])  # (None,n,hidden_dim)=(None,15,64)

        # x1 = self.layernorm1(x1)

        x2 = self.dense1(x1)  # (None,n,2*hidden_dim)=(None,15,128)

        x2 = self.dense2(x2)  # (None,n,hidden_dim)=(None,15,64)

        x2 = self.dropoout1(x2, training=training)

        features = self.add2([x1, x2])  # (None,n,hidden_dim)=(None,15,64)

        # features = self.layernorm2(features)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)=(1,15,1)

        features = features * broadcast_float_mask

        return features, score

    def build_graph(self, mask, attention_mask, idx):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask, attention_mask, training=True),
            name='mha_' + str(idx),
        )

        return model


class AgentPolicyHead(tf.keras.models.Model):
    """
    :param action_dim=5
    :param option_dim=4
    :param hidden_dim=64
    :param max_num_agents=15=n
    :return: Policy probs, (None,n,option_dim,action_dim)=(None,15,4,5)

    Model: "policy_logits"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     input_7 (InputLayer)        [(None, 15, 64)]          0

     time_distributed_7 (TimeDis  (None, 15, 192)          12480
     tributed)

     time_distributed_9 (TimeDis  (None, 15, 64)           12352
     tributed)

     time_distributed_10 (TimeDi  (None, 15, 20)           1300
     stributed)

     tf.math.multiply_4 (TFOpLam  (None, 15, 20)           0
     bda)

     time_distributed_11 (TimeDi  (None, 15, 4, 5)         0
     stributed)

    =================================================================
    Total params: 26,132
    Trainable params: 26,132
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(AgentPolicyHead, self).__init__(**kwargs)

        self.config = config

        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim * 3,
                    activation='relu',
                )
            )

        self.dropoout1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        )

        self.dense2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation='relu',
                )
            )

        self.dense3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.option_dim * self.config.action_dim,
                    activation=None,
                )
            )

        self.reshape = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Reshape(
                target_shape=(self.config.option_dim, self.config.action_dim)
            )
        )

    @tf.function
    def call(self, inputs, mask, training=True):
        # inputs: (None,n,hidden_dim)=(None,15,64)
        # mask: (None,n)=(None,15), bool
        # prob_logit=0 for dead / dummy agents

        x1 = self.dense1(inputs)  # (None,n,hidden_dim*3)

        # x1 = self.dropoout1(x1, training=training)

        x1 = self.dense2(x1)  # (None,n,hidden_dim)

        logits = self.dense3(x1)  # (None,n,option_dim * action_dim)

        # mask out dead or dummy agents by 0
        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)=(None,15,1)

        logits = logits * broadcast_float_mask  # (None,n,option_dim*action_dim)

        logits = self.reshape(logits)  # (None,n,option_dim,action_dim)

        return logits

    @tf.function
    def policy_pdf(self, policy_logits, mask):
        """
        policy_logits: (None,n,option_dim,action_dim)
        probs=0, log_probs=0 of dead / dummy agents
        """

        broadcast_float_mask = tf.expand_dims(tf.cast(mask, 'float32'), axis=-1)  # (None,n,1)
        broadcast_float_mask = \
            tf.expand_dims(tf.cast(broadcast_float_mask, 'float32'), axis=-1)  # (None,n,1,1)

        probs = tf.nn.softmax(policy_logits, axis=-1)  # (None,n,option_dim,action_dim)
        log_probs = tf.math.log(probs + 1e-5)  # (None,n,option_dim,action_dim)

        probs = probs * broadcast_float_mask  # (None,n,option_dim,action_dim)
        log_probs = log_probs * broadcast_float_mask  # (None,n,option_dim,action_dim)

        return probs, log_probs

    def sample_actions(self, policy_logits, options, mask):
        # Use only agents obs.
        # policy_logits, # (b,n,option_dim,action_dim)
        # options # (b,n), option of dead/dummy agent = option_dim+1 = 5. tf.one_hot makes this
        # to zero vector.
        # mask: # (b,n)
        """ action=5 if policyprobs=[0,0,0,0,0], that is or the dead or dummy agents """

        policy_probs, _ = self.policy_pdf(policy_logits, mask)  # (b,n,option_dim,action_dim)

        options_onehot = tf.one_hot(options, depth=self.config.option_dim)  # (b,n, option_dim)
        options_onehot_extended = tf.expand_dims(options_onehot, axis=-1)  # (b,n,option_dim,1)

        policy_probs = \
            tf.reduce_sum(policy_probs * options_onehot_extended, axis=2)  # (b,n,action_dim)

        num_agents = self.config.max_num_red_agents
        actions = []

        for i in range(num_agents):
            cdist = tfp.distributions.Categorical(probs=policy_probs[:, i, :])
            action = cdist.sample()  # (b,)
            actions.append(np.expand_dims(action.numpy(), axis=-1))

        actions = np.concatenate(actions, axis=-1)  # (b,n)

        return actions  # action=5 for the dead or dummy agents, [score1, score2]

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )  # (None,n,64)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask),
            name='policy_logits'
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

    """ global_state & feature """
    global_ch = config.global_observation_channels  # 6
    global_n_frames = config.global_n_frames

    global_state_shape = (grid_size, grid_size, global_ch * global_n_frames)  # (15,15,6)

    global_state = np.ones(shape=global_state_shape)  # (15,15,6)
    global_state = np.expand_dims(global_state, axis=0)  # (1,15,15,6)

    global_cnn = GlobalCNNModel(config=config)
    global_feature = global_cnn(global_state)  # (1,hidden_dim)=(1,256)

    """ agent observation """
    ch = config.observation_channels
    n_frames = config.n_frames

    obs_shape = (2 * fov + 1, 2 * fov + 1, ch * n_frames)  # (5,5,16)
    pos_shape = (2 * n_frames,)  # (8,)

    max_num_agents = config.max_num_red_agents

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [1, 4]
    agent_obs = {}
    agent_pos = {}

    for i in alive_agents_ids:
        agent_id = 'red_' + str(i)
        agent_obs[agent_id] = np.ones(obs_shape)
        agent_pos[agent_id] = np.ones(pos_shape) * i

    # Get padded_obs and mask
    padded_obs = make_padded_obs(max_num_agents, obs_shape,
                                 agent_obs)  # (1,n,2*fov+1,2*fov+1,ch*n_frames)

    padded_pos = make_padded_pos(max_num_agents, pos_shape, agent_pos)  # (1,n,2*n_frames)

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

    """ cnn_model """
    cnn = CNNModel(config=config)

    features_cnn = cnn([padded_obs, padded_pos], mask)  # Build, (1,n,hidden_dim)

    """ remove tf.function for summary """
    """
    cnn.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        cnn.build_graph(mask),
        to_file=dir_name + '/cnn_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ mha model """
    mha = MultiHeadAttentionModel(config=config)

    features_mha, score = mha(features_cnn, mask,
                              attention_mask)  # Build, (None,n,hidden_dim),(1,num_heads,n,n)

    """ remove tf.function for summary """
    """
    idx = 1
    mha.build_graph(mask, attention_mask, idx).summary()

    tf.keras.utils.plot_model(
        mha.build_graph(mask, attention_mask, idx),
        to_file=dir_name + '/mha_model_' + str(idx),
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ policy_model """

    policy_model = AgentPolicyHead(config=config)

    policy_logits = policy_model(features_mha, mask)
    print(f'policy_logits.shape: {policy_logits.shape}, {policy_logits}')

    """ remove tf.function for summary """
    """
    policy_model.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        policy_model.build_graph(mask),
        to_file=dir_name + '/policy_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    probs, log_probs = policy_model.policy_pdf(policy_logits, mask)
    print(f"probs: {probs}, {probs.shape}")
    print(f"log_probs: {log_probs}, {log_probs.shape}")

    """ Sample actions """
    options = np.ones((config.max_num_red_agents,)) * (config.option_dim + 1)  # (n,)
    options[1] = 0
    options[4] = 2
    options = np.expand_dims(options, axis=0)  # (1,n)

    actions = policy_model.sample_actions(policy_logits, options, mask)  # (b,n), int32
    # actions=5 for dead/dummy agents

    print('\n')
    print(probs)
    print(actions)



if __name__ == '__main__':
    main()

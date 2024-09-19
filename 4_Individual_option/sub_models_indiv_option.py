""" Sum-models for each agent """
import os.path

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from config_oc_rev import Config
from utils_transformer import make_mask, make_padded_obs
from global_models_indv_option import GlobalCNNModel


class CNNModel(tf.keras.models.Model):
    """
    # Add AutoEncoder
    :param obs_shape: (15,15,6), grid=15, ch=6, n_frames=1
    :param max_num_agents=n=15
    :return: (None,n,hidden_dim)=(None,15,256)

    Model: "cnn_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     input_1 (InputLayer)        [(None, 15, 15, 15, 6)]   0

     time_distributed (TimeDistr  (None, 15, 15, 15, 64)   448
     ibuted)

     time_distributed_1 (TimeDis  (None, 15, 7, 7, 64)     36928
     tributed)

     time_distributed_2 (TimeDis  (None, 15, 5, 5, 64)     36928
     tributed)

     time_distributed_3 (TimeDis  (None, 15, 3, 3, 64)     36928
     tributed)

     time_distributed_4 (TimeDis  (None, 15, 576)          0
     tributed)

     time_distributed_5 (TimeDis  (None, 15, 256)          147712
     tributed)

     tf.math.multiply (TFOpLambd  (None, 15, 256)          0
     a)

    =================================================================
    Total params: 258,944
    Trainable params: 258,944
    Non-trainable params: 0
    _________________________________________________________________
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
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
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

        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation=None
                )
            )

    @tf.function
    def call(self, inputs, mask):
        # inputs: (b,n,g,g,ch*n_frames)=(1,15,15,15,6)
        # mask: (b,n)=(1,15), bool

        h = self.conv0(inputs)  # (1,15,20,20,64)
        h = self.conv1(h)  # (1,15,7,7,64)
        h = self.conv2(h)  # (1,15,5,5,64)
        h = self.conv3(h)  # (1,15,3,3,64)

        h1 = self.flatten1(h)  # (1,15,576)

        features = self.dense1(h1)  # (1,15,256)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (1,15,1)

        features = features * broadcast_float_mask  # (1,15,256)

        return features

    def build_graph(self, mask):
        """ For summary & plot_model """
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames)
        )

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
    :param hidden_dim = 256

    :return: features: (None,n,hidden_dim)=(None,15,256)
             score: (None,num_heads,n,n)=(None,2,2,15)

    Model: "mha_1"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_3 (InputLayer)           [(None, 15, 256)]    0           []

     multi_head_attention (MultiHea  ((None, 15, 256),   263168      ['input_3[0][0]',
     dAttention)                     (None, 2, 15, 15))               'input_3[0][0]',
                                                                      'input_3[0][0]']

     add (Add)                      (None, 15, 256)      0           ['input_3[0][0]',
                                                                      'multi_head_attention[0][0]']

     dense_1 (Dense)                (None, 15, 512)      131584      ['add[0][0]']

     dense_2 (Dense)                (None, 15, 256)      131328      ['dense_1[0][0]']

     dropout (Dropout)              (None, 15, 256)      0           ['dense_2[0][0]']

     add_1 (Add)                    (None, 15, 256)      0           ['add[0][0]',
                                                                      'dropout[0][0]']

     tf.math.multiply_2 (TFOpLambda  (None, 15, 256)     0           ['add_1[0][0]']
     )

    ==================================================================================================
    Total params: 526,080
    Trainable params: 526,080
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
    def call(self, inputs, mask=None, training=True):
        # inputs: (None,n,hidden_dim)=(None,15,256)
        # mask: (None,n)=(None,15), bool

        float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # (None,n,1)

        attention_mask = tf.matmul(
            float_mask, float_mask, transpose_b=True
        )  # (None,n,n)

        attention_mask = tf.cast(attention_mask, 'bool')

        x, score = \
            self.mha1(
                query=inputs,
                key=inputs,
                value=inputs,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )  # (None,n,hidden_dim),(None,num_heads,n,n)=(None,15,256),(None,2,15,15)

        x1 = self.add1([inputs, x])  # (None,n,hidden_dim)=(None,15,256)

        # x1 = self.layernorm1(x1)

        x2 = self.dense1(x1)  # (None,n,hidden_dim)=(None,15,512)

        x2 = self.dense2(x2)  # (None,n,hidden_dim)=(None,15,256)

        x2 = self.dropoout1(x2, training=training)

        features = self.add2([x1, x2])  # (None,n,hidden_dim)=(None,15,256)

        # features = self.layernorm2(features)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)=(1,15,1)

        features = features * broadcast_float_mask

        return features, score

    def build_graph(self, mask, idx):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask, training=True),
            name='mha_' + str(idx),
        )

        return model


class AgentPolicyHead(tf.keras.models.Model):
    """
    :param action_dim=5
    :param option_dim=4
    :param hidden_dim=256
    :param max_num_agents=15=n
    :return: Policy probs, (None,n,option_dim,action_dim)=(None,15,4,5)

    Model: "policy_logits"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     input_1 (InputLayer)        [(None, 15, 256)]         0

     time_distributed_6 (TimeDis  (None, 15, 768)          197376
     tributed)

     time_distributed_8 (TimeDis  (None, 15, 256)          196864
     tributed)

     time_distributed_9 (TimeDis  (None, 15, 20)           5140
     tributed)

     tf.math.multiply (TFOpLambd  (None, 15, 20)           0
     a)

     time_distributed_10 (TimeDi  (None, 15, 4, 5)         0
     stributed)

    =================================================================
    Total params: 399,380
    Trainable params: 399,380
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
        # inputs: (None,n,hidden_dim)=(None,15,256)
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
        )  # (None,n,256)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask),
            name='policy_logits'
        )

        return model


class AgentQHead(tf.keras.models.Model):
    """
    :param option_dim=4
    :param hidden_dim=256
    :param max_num_agents=15=n
    :inputs: [agents_feature, global_feature]=[(None,n,hidden_dim),(None,hidden_dim)]
    :return: agent_Q_Omegas, (None,n,option_dim)=(None,15,4)

    Model: "Q_Omega_model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_4 (InputLayer)           [(None, 256)]        0           []

     tf.expand_dims (TFOpLambda)    (None, 1, 256)       0           ['input_4[0][0]']

     input_3 (InputLayer)           [(None, 15, 256)]    0           []

     tf.tile (TFOpLambda)           (None, 15, 256)      0           ['tf.expand_dims[0][0]']

     tf.concat (TFOpLambda)         (None, 15, 512)      0           ['input_3[0][0]',
                                                                      'tf.tile[0][0]']

     time_distributed_11 (TimeDistr  (None, 15, 768)     393984      ['tf.concat[0][0]']
     ibuted)

     time_distributed_12 (TimeDistr  (None, 15, 256)     196864      ['time_distributed_11[0][0]']
     ibuted)

     time_distributed_13 (TimeDistr  (None, 15, 4)       1028        ['time_distributed_12[0][0]']
     ibuted)

     tf.math.multiply_2 (TFOpLambda  (None, 15, 4)       0           ['time_distributed_13[0][0]']
     )

    ==================================================================================================
    Total params: 591,876
    Trainable params: 591,876
    Non-trainable params: 0
    __________________________________________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(AgentQHead, self).__init__(**kwargs)

        self.config = config
        self.option_dim = config.option_dim

        # For Q1
        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim * 3,
                    activation='relu',
                )
            )

        # self.dropoout1 = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        # )

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
                    units=self.option_dim,
                    activation=None,
                )
            )

    @tf.function
    def call(self, inputs, mask, training=True):
        """
        Q_Omega = 0 for dead / dummy agents
        inputs: [agents_feature, global_feature]=[(None,n,hidden_dim),(None,hidden_dim)]
        mask: (None,n)
        """

        """ mask out dead or dummy agents by 0 """
        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)

        """ Concatenate agents_feature and global_feature """
        x = inputs[0]  # (None,n,hidden_dim)
        x_global = inputs[1]  # (None, hidden_dim)

        x_global = tf.expand_dims(x_global, axis=1)  # (None,1,hidden_dim)

        mult = tf.constant([1, self.config.max_num_red_agents, 1])  # [1,n,1], (3,)
        x_global = tf.tile(x_global, mult)  # (None,n,hidden_dim)

        x = tf.concat([x, x_global], axis=-1)  # (None,n,2*hidden_dim)=(None,n,512)

        """ Agent_Q_Omega """
        x1 = self.dense1(x)  # (None,n,3*hidden_dim)=(None,n,768)
        # x1 = self.dropoout1(x1, training=training)
        x1 = self.dense2(x1)
        qs1 = self.dense3(x1)  # (None,n,option_dim)

        qs1 = qs1 * broadcast_float_mask  # (None,n,option_dim)

        return qs1

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )  # (None,n,256)

        x_global = tf.keras.layers.Input(
            shape=(self.config.hidden_dim,)
        )  # (None,256)

        model = tf.keras.models.Model(
            inputs=[x, x_global],
            outputs=self.call([x, x_global], mask),
            name='Q_Omega_model'
        )

        return model


class AgentTerminationHead(tf.keras.models.Model):
    """
    :param option_dim=4
    :param hidden_dim=256
    :param max_num_agents=15=n
    :inputs: [agents_feature, global_feature]=[(None,n,hidden_dim),(None,hidden_dim)]
    :return: agent_termination_probability, (None,n,option_dim)=(None,15,4)

    Model: "agent_termination_head"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     time_distributed_14 (TimeDi  multiple                 393984
     stributed)

     time_distributed_15 (TimeDi  multiple                 196864
     stributed)

     time_distributed_16 (TimeDi  multiple                 1028
     stributed)

    =================================================================
    Total params: 591,876
    Trainable params: 591,876
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(AgentTerminationHead, self).__init__(**kwargs)

        self.config = config
        self.option_dim = config.option_dim

        # For Q1
        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim * 3,
                    activation='relu',
                )
            )

        # self.dropoout1 = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        # )

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
                    units=self.option_dim,
                    activation='sigmoid',
                )
            )

    @tf.function
    def call(self, inputs, mask, training=False):
        """
        termination probability = 0 for dead / dummy agents
        inputs: [agents_feature, global_feature]=[(None,n,hidden_dim),(None,hidden_dim)]
        mask: (None,n)
        """

        """ mask out dead or dummy agents by 0 """
        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)

        """ Concatenate agents_feature and global_feature """
        x = inputs[0]  # (None,n,hidden_dim)
        x_global = inputs[1]  # (None, hidden_dim)

        x_global = tf.expand_dims(x_global, axis=1)  # (None,1,hidden_dim)

        mult = tf.constant([1, self.config.max_num_red_agents, 1])  # [1,n,1], (3,)
        x_global = tf.tile(x_global, mult)  # (None,n,hidden_dim)

        x = tf.concat([x, x_global], axis=-1)  # (None,n,2*hidden_dim)=(None,n,512)

        """ Agent_Q_Omega """
        x1 = self.dense1(x)  # (None,n,3*hidden_dim)=(None,n,768)
        # x1 = self.dropoout1(x1, training=training)
        x1 = self.dense2(x1)
        termination_probs = self.dense3(x1)  # (None,n,option_dim)

        termination_probs = termination_probs * broadcast_float_mask  # (None,n,option_dim)

        return termination_probs

    def sample_termination(self, termination_probs, options):
        """
        :param termination_probs: (b,n, option_dim)
        :param options: (b,n)
        :return: Bools of termination  # (b,n), ndarray
        ### For dead / dummy agents, termination=False
        """

        options_onehot = tf.one_hot(options, depth=self.option_dim)  # (b,n,4)

        y = tf.reduce_sum(termination_probs * options_onehot, axis=-1)  # (b,n)

        return np.random.uniform(size=y.shape) < y.numpy()  # (b,n), ndarray, bool


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    grid_size = config.grid_size

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

    """ cnn_model """
    cnn = CNNModel(config=config)

    features_cnn = cnn(padded_obs, mask)  # Build, (1,n,hidden_dim)

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

    features_mha, score = mha(features_cnn, mask)  # Build, (None,n,hidden_dim),(1,num_heads,n,n)

    """ remove tf.function for summary """
    """
    idx = 1
    mha.build_graph(mask, idx).summary()

    tf.keras.utils.plot_model(
        mha.build_graph(mask, idx),
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
    options[0] = 0
    options[2] = 2
    options = np.expand_dims(options, axis=0)  # (1,n)

    actions = policy_model.sample_actions(policy_logits, options, mask)  # (b,n), int32
    # actions=5 for dead/dummy agents

    print('\n')
    print(probs)
    print(actions)

    """ value_model """

    q_model = AgentQHead(config=config)

    q_omegas = q_model([features_mha, global_feature], mask)
    print(f'Q_Omegas.shape: {q_omegas.shape}')

    """ remove tf.function for summary """
    """
    q_model.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        q_model.build_graph(mask),
        to_file=dir_name + '/q_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """" Termination model """
    termination_model = AgentTerminationHead(config=config)

    termination_probs = termination_model([features_mha, global_feature], mask)  # (b,n)
    # termination_prob=0-vector for dead/dummy agents

    termination_bool = termination_model.sample_termination(termination_probs, options)  # (b,n)
    # termination_bool=False for dead/dummy agents

    termination_model.summary()


if __name__ == '__main__':
    main()

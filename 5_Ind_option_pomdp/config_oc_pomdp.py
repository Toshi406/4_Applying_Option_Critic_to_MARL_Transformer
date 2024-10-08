"""
Copied from Applying_Option_Critic_to_MARL_Transformer/4_Individual_option/config_oc_pomdp.py
"""
import numpy as np
import gym


class Config:
    def __init__(self):

        self.model_dir = 'models/oc_model_1765000/'  # newest file -> 'ls -ltr'
        # self.model_dir = None

        if self.model_dir:  # starting steps for continual training
            self.n0 = 1765000  # learner update cycles. Should be read from tensorboard
        else:
            self.n0 = 0

        # Define simulation cond.
        self.show_each_episode_result = False  # mainly for debug
        self.draw_win_distributions = False  # mainly for debug
        self.max_episodes_test_play = 50  # default=50 for training

        # Animation setting
        self.make_animation = False  # Use self.max_episodes_test_play=1

        # Time plot of a test setting
        self.make_time_plot = False  # Use self.max_episodes_test_play=1

        # POMDP params
        self.fov = 2  # default=2
        self.com = 2  # default=2

        # Define environment parameters
        self.grid_size = 15  # default=15
        self.offset = 0  # blue-team offset from edges

        # When agents are tested in larger battlefield, the global model need to be built
        # with grid=size at the training
        self.global_grid_size = 15

        # Define gym spaces
        self.action_dim = 5
        self.action_space = gym.spaces.Discrete(self.action_dim)

        self.option_dim = 4

        observation_low = 0.
        observation_high = 1.
        self.observation_channels = 4
        self.n_frames = 4
        self.observation_space = \
            gym.spaces.Box(low=observation_low,
                           high=observation_high,
                           shape=(2 * self.fov + 1,
                                  2 * self.fov + 1,
                                  self.observation_channels)
                           )

        self.global_observation_channels = 6
        self.global_n_frames = 1

        # epsilon-greedy of policy over options
        self.epsilon = 0.1

        # Neural nets parameters
        self.hidden_dim = 64
        self.key_dim = 64
        self.num_heads = 2

        self.dropout_rate = 0.2  # default=0.2, (Dropout is not used.)

        # Training parameters
        self.actor_rollout_steps = 1  # default=1
        self.num_update_cycles = 100000000
        self.batch_size = self.actor_rollout_steps  # Should be 1 for option-critic

        # self.num_minibatchs = 3  # default=3
        self.tau = 0.1  # Soft update of target network
        self.gamma = 0.96
        self.max_steps = 100  # Default=100. 200 for robustness

        self.learning_rate = 5e-5  # Default=5e-5

        self.entropy_coef = 0.01  # For entropy_loss, Default=0.01
        self.termination_reg = 0.01  # For termination regularizer, Default = 0.01

        self.q_loss_coef = 5.0
        self.policy_loss_coef = 1.0
        self.termination_loss_coef = 1.0
        self.loss_coef = 1.0

        # Define Lanchester simulation parameters
        self.threshold = 5.0  # min of forces R & B
        self.log_threshold = np.log(self.threshold)
        self.mul = 2.0  # Minimum platoon force = threshold * mul
        self.dt = .2  # Default=.2

        # Define possible agent parameters
        self.agent_types = ('platoon', 'company')
        self.agent_forces = (50, 150)

        # Define possible red / blue agent parameters
        self.red_platoons = (3, 10)  # num range of red platoons, default=(3,10)
        self.red_companies = (2, 5)  # num range of red companies, default=(2,5)

        self.blue_platoons = (3, 10)  # num range of blue platoons, default=(3,10)
        self.blue_companies = (2, 5)  # num range of blue companies, default=(2,5)

        self.efficiencies_red = (0.3, 0.5)  # range
        self.efficiencies_blue = (0.3, 0.5)

        # For paddiing of multi-agents, *3 for adding red agents, default:*1
        self.max_num_red_agents = (self.red_platoons[1] + self.red_companies[1])

        # Red team TBD parameters
        self.R0 = None  # initial total force, set in 'generate_red_team'
        self.log_R0 = None
        self.num_red_agents = None  # set in 'define_red_team'
        self.num_red_platoons = None
        self.num_red_companies = None

        # Blue team TBD parameters
        self.B0 = None  # initial total force, set in 'generate_blue_team'
        self.log_B0 = None
        self.num_blue_agents = None  # set in 'define_blue_team'
        self.num_blue_platoons = None
        self.num_blue_companies = None

    def define_blue_team(self):
        """
        Called from reset
            self.num_blue_agents, self.num_blue_platoons, self.num_blue_companies
            will be allocated.
        """
        self.num_blue_platoons = \
            np.random.randint(
                low=self.blue_platoons[0],
                high=self.blue_platoons[1] + 1)

        self.num_blue_companies = \
            np.random.randint(
                low=self.blue_companies[0],
                high=self.blue_companies[1] + 1)

        self.num_blue_agents = self.num_blue_platoons + self.num_blue_companies

    def define_red_team(self):
        """
        Called from reset
            self.num_red_agents, self.num_red_platoons, self.num_red_companies
            will be allocated.
        """
        self.num_red_platoons = \
            np.random.randint(
                low=self.red_platoons[0],
                high=self.red_platoons[1] + 1)

        self.num_red_companies = \
            np.random.randint(
                low=self.red_companies[0],
                high=self.red_companies[1] + 1)

        self.num_red_agents = self.num_red_platoons + self.num_red_companies

    def reset(self):
        """
        Generate new config for new episode
        """
        self.define_blue_team()
        self.define_red_team()


if __name__ == '__main__':
    config = Config()

    for _ in range(3):
        config.reset()

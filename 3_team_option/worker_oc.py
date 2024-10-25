import ray
import numpy as np
import tensorflow as tf

from collections import deque

from battlefield_strategy_team_reward_global_state import BattleFieldStrategy

from models_oc_rev import EpsGreedyOverOptions, Individual_OptionCritic_Model
from utils_gnn import get_alive_agents_ids
from utils_transformer import make_mask, make_padded_obs
from utils_oc import get_option


# @ray.remote(num_cpus=1, num_gpus=0)  # cloud使用時
@ray.remote
class Worker:
    def __init__(self, worker_id):
        """
        batch_size=t_max
        """
        self.worker_id = worker_id

        self.env = BattleFieldStrategy()
        self.gamma = self.env.config.gamma
        self.batch_size = self.env.config.batch_size
        self.n_frames = self.env.config.n_frames

        self.action_space_dim = self.env.action_space.n

        self.global_n_frames = self.env.config.global_n_frames

        # Instantiate models
        self.policy_over_options = EpsGreedyOverOptions(config=self.env.config)
        self.oc_model = Individual_OptionCritic_Model(config=self.env.config)

        self.obs_shape = (self.env.config.grid_size,
                          self.env.config.grid_size,
                          self.env.config.observation_channels * self.n_frames)

        # Initialize environment
        ### The followings are reset in 'reset_states'
        self.frames = None  # For each agent in env
        self.states = None
        self.prev_actions = None

        self.alive_agents_ids = None  # For all agents, including dummy ones
        self.padded_states = None
        self.padded_prev_actions = None
        self.mask = None

        self.global_frames = None  # Frame stack for global state
        self.global_state = None

        self.option = None
        self.termination_probs = None
        self.agents_q_omega = None
        self.agents_policy_logit = None

        self.individual_return = None
        self.team_return = None

        self.step = None

        ### Initialize above Nones
        observations, global_observation = self.env.reset()
        self.reset_states(observations, global_observation)

        self.oc_model([self.padded_states, self.global_state], self.mask)  # build

    def reset_states(self, observations, global_observation):
        # TODO prev_actions
        """
        alive_agents_ids: list of alive agent id

        # For agents in Env
             each agent stacks observations n-frames in channel-dims
             -> observations[red.id]: (grid_size,grid_size,channels)

             -> generate deque of length=n_frames
             self.frames[red.id]: deque[(grid_size,grid_size,channels),...]

             -> transform to states
             states[red.id]: (grid_size,grid_size,channels*n_frames)

             self.prev_actions[red.id]: int (TODO)
        """

        self.frames = {}
        self.states = {}
        self.prev_actions = {}

        for red in self.env.reds:
            # all reds are alive when reset

            self.frames[red.id] = deque([observations[red.id]] * self.n_frames,
                                        maxlen=self.n_frames)
            # [(grid_size,grid_size,channels),...,(grid_size,grid_size,channels)]

            self.states[red.id] = np.concatenate(self.frames[red.id], axis=2).astype(np.float32)
            # (grid_size,grid_size,channels*n_frames)

            # self.prev_actions[red.id] = 0

        # alive_agents_ids: list of alive agent id, [int,...], len=num_alive_agents
        self.alive_agents_ids = get_alive_agents_ids(env=self.env)

        # Get padded observations ndarray for all agents, including dead and dummy agents
        self.padded_states = \
            make_padded_obs(max_num_agents=self.env.config.max_num_red_agents,
                            obs_shape=self.obs_shape,
                            raw_obs=self.states)  # (1,n,g,g,ch*n_frames)

        # Get global_state
        self.global_frames = deque([global_observation] * self.global_n_frames,
                                   maxlen=self.global_n_frames)
        global_state = np.concatenate(self.global_frames, axis=2).astype(np.float32)
        # (g,g,global_ch*global_n_frames)
        self.global_state = np.expand_dims(global_state, axis=0)
        # (1,g,g,global_ch*global_n_frames)

        # Get mask for the padding
        self.mask = make_mask(alive_agents_ids=self.alive_agents_ids,
                              max_num_agents=self.env.config.max_num_red_agents)  # (1,n)

        # get option
        self.termination_probs, self.agents_q_omega, self.agents_policy_logit, _ = \
            self.oc_model([self.padded_states, self.global_state], self.mask)
        # (b,option_dim),(b,n,option_dim),(b,n,option_dim,action_dim)

        self.option = get_option(self.agents_q_omega, self.mask, self.policy_over_options)

        # reset episode variables
        self.individual_return = 0
        self.team_return = 0
        self.step = 0

    def rollout_and_collect_trajectory(self, weights):
        """
        0. Global policyの重みをコピー
        1. Rolloutして、batch_size分のデータを収集
        """

        """ 
        0. Global policyの重みをコピー 
        """
        self.oc_model.set_weights(weights=weights)

        """ 
        1. Rolloutして、batch_size分のデータを収集
            batch_size = sequence_length = b = 1  ### batch_size should be 1 for option-critic
            max_num_red_agents = n
            trajectory["s"]: (b,n,g,g,ch*n_frames)
            trajectory["a"]: (b,n), np.int32
            trajectory["r"]: (b,n)
            trajectory["dones"]: (b,n), bool
            trajectory["s2"]: next_states, (b,n,g,g,ch*n_frames)
            trajectory["mask"]: (b,n), bool
            trajectory["mask2"]: next_mask, (b,n), bool
            
            trajectory["global_s"]: (b,g,g,global_ch*global_n_frames)
            trajectory["global_s2"]: (b,g,g,global_ch*global_n_frames)
            trajectory["option"]: (b,), np.int32
            trajectory["team_r"]: (b,), np.float32
            trajectory["team_done"]: (b,), bool
        """

        trajectory = self._rollout()

        return trajectory

    def _rollout(self):
        """
        Rolloutにより、t_start<=t<t_max 間(batch_size間)の
        {s,a,s',r,done,mask,mask',global_s,global_s2,option,team_r,team_done}を取得
        """
        # 1. 初期化
        trajectory = {}
        trajectory["s"] = []
        trajectory["a"] = []
        trajectory["r"] = []
        trajectory["s2"] = []
        trajectory["dones"] = []
        trajectory["mask"] = []
        trajectory["mask2"] = []  # next_mask

        trajectory["global_s"] = []
        trajectory["global_s2"] = []
        trajectory["option"] = []
        trajectory["team_r"] = []
        trajectory["team_done"] = []

        # 2. Rollout実施
        for i in range(self.batch_size):
            # acts: action=-1 for the dead or dummy agents.  (1,n)
            acts = self.oc_model.agents_policy_head.sample_actions(
                self.agents_policy_logit, self.option, self.mask)

            # get alive_agents & all agents actions.
            # * padded_actions: action=-1 for dead or dummy agents for
            #       utilizing tf.one_hot(-1) is zero vector
            # * actions['red_a'], a=alive agent id
            actions = {}  # For alive agents
            padded_actions = - np.ones((1, self.env.config.max_num_red_agents))  # (1,n)

            for idx in self.alive_agents_ids:
                agent_id = 'red_' + str(idx)
                actions[agent_id] = acts[0, idx]

                padded_actions[0, idx] = actions[agent_id]

            # One step of Lanchester simulation, for alive agents in env
            # next_observations: {agent_id:(g,g,ch)}
            # rewards: individual rewards
            # dones: individual dones
            # reward: team reward, float64
            # done team done, bool
            # next_global_observation: (g,g,global_ch)
            next_obserations, rewards, dones, infos, reward, done, next_global_observation = \
                self.env.step(actions)

            # Make next_agents_states, next_agents_adjs, and next_alive_agents_ids,
            # including dummy ones
            next_alive_agents_ids = get_alive_agents_ids(env=self.env)

            ### For alive agents in env
            next_states = {}

            for idx in next_alive_agents_ids:
                agent_id = 'red_' + str(idx)

                self.frames[agent_id].append(
                    next_obserations[agent_id]
                )  # append (g,g,ch) to deque

                next_states[agent_id] = np.concatenate(
                    self.frames[agent_id], axis=2
                ).astype(np.float32)  # (g,g,ch*n_frames)

            # Get padded next observations ndarray of all agent
            next_padded_states = \
                make_padded_obs(
                    max_num_agents=self.env.config.max_num_red_agents,
                    obs_shape=self.obs_shape,
                    raw_obs=next_states
                )  # (1,n,g,g,ch*n_frames)

            # Get next_global_state
            self.global_frames.append(next_global_observation)
            # append (g,g,global_ch)

            next_global_state = np.concatenate(self.global_frames, axis=2).astype(np.float32)
            # (g,g,global_ch*global_n_frames)

            next_global_state = np.expand_dims(next_global_state, axis=0)
            # (1,g,g,global_ch*global_n_frames)

            # Get next mask for the padding
            next_mask = \
                make_mask(
                    alive_agents_ids=next_alive_agents_ids,
                    max_num_agents=self.env.config.max_num_red_agents
                )  # (1,n)

            # 終了判定
            if self.step > self.env.config.max_steps:

                for idx in self.alive_agents_ids:
                    agent_id = 'red_' + str(idx)
                    dones[agent_id] = True

                dones['all_dones'] = True

            # agents_rewards and agents_dones, including dead and dummy ones
            # reward = 0 and done=True for dead or dummy agents.
            agents_rewards = []
            agents_dones = []

            for idx in range(self.env.config.max_num_red_agents):
                if idx in self.alive_agents_ids:
                    agent_id = 'red_' + str(idx)
                    agents_rewards.append(float(rewards[agent_id]))
                    agents_dones.append(dones[agent_id])
                else:
                    agents_rewards.append(0.0)
                    agents_dones.append(True)

            # if len(agents_rewards) != self.env.config.max_num_red_agents:
            #     raise ValueError()

            # if len(agents_dones) != self.env.config.max_num_red_agents:
            #     raise ValueError()

            # Update returns
            self.individual_return += np.sum(agents_rewards)
            self.team_return += reward  # float

            # list -> ndarray
            padded_rewards = np.stack(agents_rewards, axis=0)  # (n,)
            padded_rewards = np.expand_dims(padded_rewards, axis=0)  # (1,n)

            padded_dones = np.stack(agents_dones, axis=0)  # (n,), bool
            padded_dones = np.expand_dims(padded_dones, axis=0)  # (1,n)

            # alive_agents_ids = np.array(self.alive_agents_ids, dtype=object)  # (a,), object
            # alive_agents_ids = np.expand_dims(alive_agents_ids, axis=0)  # (1,a)

            trajectory["s"].append(self.padded_states)  # append (1,n,g,g,ch*n_frames)
            trajectory["a"].append(padded_actions)  # append (1,n)
            trajectory["r"].append(padded_rewards)  # append (1,n)
            trajectory["s2"].append(next_padded_states)  # append (1,n,g,g,ch*n_frames)
            trajectory["dones"].append(padded_dones)  # append (1,n)
            trajectory["mask"].append(self.mask)  # append (1,n)
            trajectory["mask2"].append(next_mask)  # append (1,n)

            trajectory["global_s"].append(self.global_state)
            # append(1,g,g,global_ch*global_n_frames)
            trajectory["global_s2"].append(next_global_state)
            # append(1,g,g,global_ch*global_n_frames)
            trajectory["option"].append(self.option)  # append (1,)
            trajectory["team_r"].append(np.array([reward]))  # append (1,), float
            trajectory["team_done"].append(np.array([done]))  # append (1,), bool

            # Check termination
            next_termination_probs, next_agents_q_omega, next_agents_policy_logit, scores = \
                self.oc_model([next_padded_states, next_global_state], next_mask)
            # (b,option_dim),(b,n,option_dim),(b,n,option_dim,action_dim),
            # [(b,num_heads,n,n),(b,num_heads,n,n)]

            next_termination = self.oc_model.termination_head.sample_termination(
                next_termination_probs, self.option)  # (1,1), bool

            if next_termination[0, 0]:  # bool
                self.option = get_option(next_agents_q_omega, next_mask,
                                         self.policy_over_options)  # (1,), ndarray

            if dones['all_dones']:
                # print(f'episode reward = {self.episode_return}')
                observations, global_observation = self.env.reset()
                self.reset_states(observations, global_observation)
            else:
                self.alive_agents_ids = next_alive_agents_ids
                self.padded_states = next_padded_states
                self.mask = next_mask

                self.global_state = next_global_state

                self.termination_probs = next_termination_probs
                self.agents_q_omega = next_agents_q_omega
                self.agents_policy_logit = next_agents_policy_logit

                self.step += 1

        trajectory["s"] = np.concatenate(trajectory["s"], axis=0).astype(np.float32)
        # (b,n,g,g,ch*n_frames)
        trajectory["a"] = np.concatenate(trajectory["a"], axis=0).astype(np.int32)
        # (b,n), np.int32
        trajectory["r"] = np.concatenate(trajectory["r"], axis=0).astype(np.float32)
        # (b,n)
        trajectory["s2"] = np.concatenate(trajectory["s2"], axis=0).astype(np.float32)
        # (b,n,g,g,ch*n_frames)
        trajectory["dones"] = np.concatenate(trajectory["dones"], axis=0).astype(bool)
        # (b,n), bool
        trajectory["mask"] = np.concatenate(trajectory["mask"], axis=0).astype(bool)
        # (b,n), bool
        trajectory["mask2"] = np.concatenate(trajectory["mask2"], axis=0).astype(bool)
        # (b,n), bool

        trajectory["global_s"] = np.concatenate(trajectory["global_s"], axis=0).astype(np.float32)
        # (b, g, g, global_ch * global_n_frames)
        trajectory["global_s2"] = np.concatenate(trajectory["global_s2"], axis=0).astype(np.float32)
        # (b, g, g, global_ch * global_n_frames)
        trajectory["option"] = np.concatenate(trajectory["option"], axis=0).astype(np.int32)
        # (b,), np.int32
        trajectory["team_r"] = np.concatenate(trajectory["team_r"], axis=0).astype(np.float32)
        # (b,), np.float32
        trajectory["team_done"] = np.concatenate(trajectory["team_done"], axis=0).astype(bool)
        # (b,), bool

        return trajectory

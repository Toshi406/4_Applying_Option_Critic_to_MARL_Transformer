import shutil
import ray
import numpy as np
import tensorflow as tf
import json
import os
import time

from pathlib import Path

from battlefield_strategy_team_reward_global_state import BattleFieldStrategy

from models_oc_rev import EpsGreedyOverOptions, Individual_OptionCritic_Model
from utils_transformer import make_mask, make_padded_obs
from utils_oc import get_option

from worker_oc import Worker
from tester_oc import Tester


def write_config(config):
    """
    Save Training conditions
    """
    config_list = {
        'max_episodes_test_play': config.max_episodes_test_play,
        'grid_size': config.grid_size,
        'offset': config.offset,

        'action_dim': config.action_dim,
        'observation_channels': config.observation_channels,
        'n_frames': config.n_frames,

        'option_dim': config.option_dim,
        'global_observation_channels': config.global_observation_channels,
        'global_n_frames': config.global_n_frames,

        'hidden_dim': config.hidden_dim,
        'key_dim': config.key_dim,
        'num_heads': config.num_heads,
        'dropout_rate': config.dropout_rate,

        'actor_rollout_steps': config.actor_rollout_steps,
        'num_update_cycles': config.num_update_cycles,
        'batch_size': config.batch_size,
        # 'num_minibatchs': config.num_minibatchs,

        'tau': config.tau,
        'gamma': config.gamma,

        'max_steps': config.max_steps,

        'learning_rate': config.learning_rate,

        'entropy_coef': config.entropy_coef,
        'q_loss_coef': config.q_loss_coef,
        'policy_loss_coef': config.policy_loss_coef,
        'termination_loss_coef': config.termination_loss_coef,
        'loss_coef': config.loss_coef,

        'termination_reg': config.termination_reg,

        'threshold': config.threshold,
        'mul': config.mul,
        'dt': config.dt,

        'agent_types': config.agent_types,
        'agent_forces': config.agent_forces,

        'red_platoons': config.red_platoons,
        'red_companies': config.red_companies,

        'blue_platoons': config.blue_platoons,
        'blue_companies': config.blue_companies,

        'efficiencies_red': config.efficiencies_red,
        'efficiencies_blue': config.efficiencies_blue,

        'max_num_red_agents': config.max_num_red_agents,
    }

    dir_save = './result'
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)

    with open(dir_save + '/training_conditions.json', 'w') as f:
        json.dump(config_list, f, indent=5)


def update_targets(net, target_net, tau):
    """ Soft update target_net """
    weights = net.get_weights()
    target_weights = target_net.get_weights()
    target_weights = update_target_weights(weights, target_weights, tau)
    target_net.set_weights(target_weights)
    return target_net


def update_target_weights(weights, target_weights, tau):
    for w in range(len(target_weights)):
        target_weights[w] = tau * weights[w] + (1. - tau) * target_weights[w]
    return target_weights


def learn(num_workers=8, is_debug=False):
    if is_debug:
        print("Debug mode starts. May cause ray memory error.")
    else:
        print("Execution mode starts")

    ray.init(local_mode=is_debug, ignore_reinit_error=True)

    logdir = Path(__file__).parent / "log"

    summary_writer = tf.summary.create_file_writer(logdir=str(logdir))

    start = time.time()
    history = []

    # Make result dir
    resultdir = Path(__file__).parent / "result"
    if resultdir.exists():
        shutil.rmtree(resultdir)

    """ Instantiate environment """
    env = BattleFieldStrategy()
    write_config(env.config)
    action_space = env.action_space.n

    """ Instantiate & Build policies """
    grid_size = env.config.grid_size

    global_n_frames = env.config.global_n_frames
    global_ch = env.config.global_observation_channels
    global_obs_shape = (grid_size, grid_size, global_ch * global_n_frames)

    ch = env.config.observation_channels
    n_frames = env.config.n_frames
    obs_shape = (grid_size, grid_size, ch * n_frames)

    max_num_agents = env.config.max_num_red_agents

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]
    agent_obs = {}

    for i in alive_agents_ids:
        agent_id = 'red_' + str(i)
        agent_obs[agent_id] = np.ones(obs_shape)

    # Get padded_obs, global_obs, and mask for build
    padded_obs = make_padded_obs(max_num_agents, obs_shape, agent_obs)  # (1,n,g,g,ch*n_frames)
    global_obs = np.ones(global_obs_shape)  # (g,g,global_ch*global_n_frames)
    global_state = np.expand_dims(global_obs, axis=0)  # (1,g,g,global_ch*global_n_frames)

    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    # Instantiate models
    policy_over_options = EpsGreedyOverOptions(config=env.config)
    oc_model = Individual_OptionCritic_Model(config=env.config)
    target_oc_model = Individual_OptionCritic_Model(config=env.config)

    # Build models
    oc_model([padded_obs, global_state], mask)  # build
    target_oc_model([padded_obs, global_state], mask)  # build

    oc_model.summary()

    """ Load model if necessary """
    if env.config.model_dir:
        oc_model.load_weights(env.config.model_dir)

    """ Instantiate optimizer """
    optimizer = tf.keras.optimizers.Adam(learning_rate=env.config.learning_rate)

    """ Instantiate workers """
    workers = [Worker.remote(worker_id=i) for i in range(num_workers)]

    """ Instantiate tester """
    tester = Tester.remote()

    """ get the weights of oc_model, and copy them to the targetand  """
    weights = oc_model.get_weights()

    target_oc_model.set_weights(weights)

    """ starts tester process """
    test_in_progress = tester.test_play.remote(weights)

    update_cycles = env.config.n0 + 1

    test_cycles = update_cycles

    while update_cycles <= env.config.num_update_cycles:
        """ Execute worker process, and get trajectory as list """
        trajectories = ray.get(
            [worker.rollout_and_collect_trajectory.remote(weights) for worker in workers])

        """ Reshape states, actions, masks, discounted_returns 
            batch_size = 1 for option-critic    
            w = num_workers, b=env.config.batch_size=worker_rollout_steps=1
            
            states: (w*b, n, g, g, ch*n_frames)
            next_state: (w*b, n, g, g, ch*n_frames)
            actions: (w*b, n), np.int32
            rs: (w*b, n),
            dones: (w*b, n), bool
            masks: (w*b, n), bool
            next_masks: (w*b, n), bool
            
            global_states: (w*b, g, g, global_ch*global_n_frames)
            next_global_states: (w*b, g, g, global_ch*global_n_frames)
            options: (w*b,), np.int32
            team_rs: (w*b,)
            team_dones: (w*b,), bool
        """
        # Make lists
        (states, next_states, actions, rs, dones, masks, next_masks,
         global_states, next_global_states, options, team_rs, team_dones) = \
            [], [], [], [], [], [], [], [], [], [], [], []

        for i in range(num_workers):
            states.append(trajectories[i]["s"])  # append (b,n,g,g,ch*n_frames)
            next_states.append(trajectories[i]["s2"])  # append (b,n,g,g,ch*n_frames)
            actions.append(trajectories[i]["a"])  # append (b,n)
            rs.append(trajectories[i]["r"])  # append (b,n)
            dones.append(trajectories[i]["dones"])  # append (b,n), bool
            masks.append(trajectories[i]["mask"])  # append (b,n)
            next_masks.append(trajectories[i]["mask2"])  # append (b,n)

            global_states.append(trajectories[i]["global_s"])
            # append (b,g,g,global_ch*global_n_frames)
            next_global_states.append(trajectories[i]["global_s2"])
            # append (b,g,g,global_ch*global_n_frames)
            options.append(trajectories[i]["option"])  # append (b,), int
            team_rs.append(trajectories[i]["team_r"])  # append (b,)
            team_dones.append(trajectories[i]["team_done"])  # append (b,), bool

        # lists -> np.array
        states = np.array(states, dtype=np.float32)  # (w,b,n,g,g,ch*n_frames)
        next_states = np.array(next_states, dtype=np.float32)  # (w,b,n,g,g,ch*n_frames)
        actions = np.array(actions, dtype=np.int32)  # (w,b,n), np.int32
        rs = np.array(rs, dtype=np.float32)  # (w,b,n)
        dones = np.array(dones, dtype=bool)  # (w,b,n),bool
        masks = np.array(masks, dtype=bool)  # (w,b,n) bool
        next_masks = np.array(next_masks, dtype=bool)  # (w,b,n) bool

        global_states = np.array(global_states, dtype=np.float32)
        # (w,b,g,g,global_ch*global_n_frames)
        next_global_states = np.array(next_global_states, dtype=np.float32)
        # (w,b,g,g,global_ch*global_n_frames)
        options = np.array(options, dtype=np.int32)  # (w,b)
        team_rs = np.array(team_rs, dtype=np.float32)  # (w,b)
        team_dones = np.array(team_dones, dtype=bool)  # (w,b)

        # reshape to batch_size=w*b
        batch_size = num_workers * env.config.batch_size  # w*b
        states = states.reshape([batch_size, max_num_agents, grid_size, grid_size, ch * n_frames])
        next_states = next_states.reshape(
            [batch_size, max_num_agents, grid_size, grid_size, ch * n_frames])
        actions = actions.reshape([batch_size, max_num_agents])
        rs = rs.reshape([batch_size, max_num_agents])
        dones = dones.reshape([batch_size, max_num_agents])
        masks = masks.reshape([batch_size, max_num_agents])
        next_masks = next_masks.reshape([batch_size, max_num_agents])

        global_states = global_states.reshape(
            [batch_size, grid_size, grid_size, global_ch * global_n_frames])
        next_global_states = next_global_states.reshape(
            [batch_size, grid_size, grid_size, global_ch * global_n_frames])
        options = options.reshape([batch_size, ])
        team_rs = team_rs.reshape([batch_size, ])
        team_dones = team_dones.reshape([batch_size, ])

        """ Update {Q_Omega_i}, i=1:n, """

        actions_onehot = \
            tf.one_hot(actions, depth=env.config.action_dim, axis=-1)  # (b,n,action_dim)
        options_onehot = tf.one_hot(options, depth=env.config.option_dim, axis=-1)  # (b,option_dim)
        options_onehot_extended_1 = tf.expand_dims(options_onehot, axis=1)  # (b,1,option_dim)
        options_onehot_extended_2 = tf.expand_dims(options_onehot_extended_1, axis=-1)
        # (b,1,option_dim,1)

        float_masks = tf.cast(masks, tf.float32)  # (b,n)
        float_masks_extended = tf.expand_dims(float_masks, axis=-1)  # (b,n,1)
        float_next_masks = tf.cast(next_masks, tf.float32)  # (b,n)
        float_next_masks_extended = tf.expand_dims(float_next_masks, axis=-1)  # (b,n,1)

        float_dones = tf.cast(dones, tf.float32)  # (b,n)

        num_alive_agents = tf.reduce_sum(tf.cast(masks, tf.float32), axis=-1)  # (b,)

        """ USe target_network beta(s') & q_omega(s',ω) to get gt """
        betas_dash, target_q_omegas, _, _ = \
            target_oc_model([next_states, next_global_states], next_masks)
        # (b,option_dim), (b,n,option_dim)

        betas_dash = tf.reduce_sum(betas_dash * options_onehot, axis=-1, keepdims=True)  # (b,1)
        betas_dash = tf.tile(betas_dash, tf.constant([1, max_num_agents], tf.int32))  # (b,n)

        target_Q_Omegas = target_q_omegas * options_onehot_extended_1  # (b,n,option_dim)
        target_Q_Omegas = tf.reduce_sum(target_Q_Omegas, axis=-1)  # (b,n)

        target_Q_Omega_max = \
            get_q_omega_max(target_q_omegas, env, float_next_masks_extended)  # (b,n)

        gt = rs + env.config.gamma * (1. - float_dones) * \
             ((1. - betas_dash) * target_Q_Omegas + betas_dash * target_Q_Omega_max)  # (b,n)

        with tf.GradientTape() as tape:
            # termination_probs: (b,option_dim)
            # aents_q_omega: (b,n,option_dim)
            # agents_policy_logits: (b,n,option_dim,action_dim)
            termination_probs, agents_q_omega, agents_policy_logit, _ = \
                oc_model([states, global_states], masks)

            ### value_loss, {Q^i(s^i,ω)}, i=1:n,

            Q_Omegas = agents_q_omega * options_onehot_extended_1  # (b,n,option_dim)
            Q_Omegas = tf.reduce_sum(Q_Omegas, axis=-1)  # (b,n)

            Q_Omegas_max = get_q_omega_max(agents_q_omega, env, float_masks_extended)  # (b,n)

            td_errors = gt - Q_Omegas  # (b,n)
            td_errors = td_errors * float_masks  # (b,n)

            q_loss = tf.reduce_sum(tf.math.square(td_errors), axis=-1) / num_alive_agents  # (b,)
            q_loss = tf.reduce_mean(q_loss)

            ### policy_loss, {π^i(a^i|s^i)}, i=1:n,
            policy_probs, log_policy = \
                oc_model.agents_policy_head.policy_pdf(agents_policy_logit, masks)
            # (b,n,option_dim,action_dim), (b,n,option_dim,action_dim)

            policy_probs = policy_probs * options_onehot_extended_2  # (b,n,option_dim,action_dim)
            policy_probs = tf.reduce_sum(policy_probs, axis=2)  # (b,n,action_dim)

            log_policy = log_policy * options_onehot_extended_2  # (b,n,option_dim,action_dim)
            log_policy = tf.reduce_sum(log_policy, axis=2)  # (b,n,action_dim)

            entropy = -tf.einsum('ijk,ijk->ij', policy_probs, log_policy)  # (b,n)
            entropy = entropy * float_masks  # (b,n)

            log_policy_probs = log_policy * actions_onehot  # (b,n,action_dim)
            log_policy_probs = tf.reduce_sum(log_policy_probs, axis=-1)  # (b,n)

            policy_obj = log_policy_probs * tf.stop_gradient(gt - Q_Omegas)  # (b,n)
            policy_obj = policy_obj * float_masks  # (b,n)

            policy_obj = \
                tf.reduce_sum(
                    policy_obj + env.config.entropy_coef * entropy, axis=-1) / num_alive_agents
            # (b,)

            policy_loss = -tf.reduce_mean(policy_obj)

            ### termination loss β_ω(s)
            termination_reg = env.config.termination_reg

            advantage = Q_Omegas - Q_Omegas_max  # (b,n)
            advantage = advantage * float_masks  # (b,n)
            advantage = tf.reduce_sum(advantage, axis=-1) / num_alive_agents  # (b,)
            advantage = tf.stop_gradient(advantage)  # (b,)

            beta = termination_probs * options_onehot  # (b,option_dim)
            beta = tf.reduce_sum(beta, axis=-1)  # (b,)

            termination_loss = tf.reduce_mean(beta * (advantage + termination_reg))

            ### total loss
            q_coef = env.config.q_loss_coef
            policy_coef = env.config.policy_loss_coef
            termination_coef = env.config.termination_loss_coef
            loss_coef = env.config.loss_coef

            loss = q_coef * q_loss + policy_coef * policy_loss + termination_coef * termination_loss
            loss = loss_coef * loss

        grads = tape.gradient(loss, oc_model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        optimizer.apply_gradients(zip(grads, oc_model.trainable_variables))

        info = {
            "value_loss": q_coef * q_loss,
            "policy_loss": policy_coef * policy_loss,
            "termination_loss": termination_coef * termination_loss,
            "entropy_loss": -1 * policy_coef * env.config.entropy_coef * tf.reduce_mean(entropy),
            "advantage": tf.reduce_mean(advantage),
            "loss": loss,
        }

        # Soft update of the target
        target_oc_model = update_targets(oc_model, target_oc_model, tau=env.config.tau)

        if update_cycles % 100 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("policy_loss", info["policy_loss"], step=update_cycles)
                tf.summary.scalar("value_loss", info["value_loss"], step=update_cycles)
                tf.summary.scalar("termination_loss", info["termination_loss"], step=update_cycles)
                tf.summary.scalar("entropy_loss", info["entropy_loss"], step=update_cycles)
                tf.summary.scalar("advantage", info["advantage"], step=update_cycles)
                tf.summary.scalar("loss", info["loss"], step=update_cycles)

        if update_cycles % 5000 == 0:
            model_name = "oc_model_" + str(update_cycles)
            oc_model.save_weights('models/' + model_name + '/')

        update_cycles += 1

        # get updated weights
        weights = oc_model.get_weights()

        finished_tester, _ = ray.wait([test_in_progress], timeout=0)
        if finished_tester:
            result = ray.get(finished_tester[0])

            print(f"test_cycles={test_cycles}, test_score={result['individual_return']}, "
                  f"episode_len={result['episode_lens']}")
            history.append((test_cycles, result['individual_return']))

            with summary_writer.as_default():
                tf.summary.scalar(
                    "mean_individual_return of tests", result['individual_return'],
                    step=test_cycles)
                tf.summary.scalar(
                    "mean_team_return of tests", result['team_return'],
                    step=test_cycles)
                tf.summary.scalar(
                    "mean_episode_len of tests", result['episode_lens'], step=test_cycles)
                tf.summary.scalar(
                    "mean_option_switches", result['option_switches'], step=test_cycles)

                tf.summary.scalar(
                    "mean_num_alive_red_ratio", result['num_alive_reds_ratio'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_red_platoon",
                    result['num_alive_red_platoon'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_red_company",
                    result['num_alive_red_company'], step=test_cycles)
                tf.summary.scalar(
                    "mean_remaining_red_effective_force_ratio",
                    result['remaining_red_effective_force_ratio'], step=test_cycles)

                tf.summary.scalar(
                    "mean_num_alive_blue_ratio", result['num_alive_blues_ratio'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_blue_platoon",
                    result['num_alive_blue_platoon'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_blue_company",
                    result['num_alive_blue_company'], step=test_cycles)
                tf.summary.scalar(
                    "mean_remaining_blue_effective_force_ratio",
                    result['remaining_blue_effective_force_ratio'], step=test_cycles)

                tf.summary.scalar(
                    "num_red_win", result['num_red_win'], step=test_cycles)
                tf.summary.scalar(
                    "num_blue_win", result['num_blue_win'], step=test_cycles)
                tf.summary.scalar(
                    "num_draw", result['draw'], step=test_cycles)
                tf.summary.scalar(
                    "num_no_contest", result['no_contest'], step=test_cycles)

            test_cycles = update_cycles
            test_in_progress = tester.test_play.remote(weights)

    else:
        ray.shutdown()


def get_q_omega_max(agents_q_omega, env, float_masks_extended):
    sum_q_omegas = \
        tf.reduce_sum(agents_q_omega * float_masks_extended, axis=1)  # (b,option_dim)

    max_omegas = np.argmax(sum_q_omegas.numpy(), axis=-1)  # (b,)
    max_omegas_onehot = np.eye(env.config.option_dim)[max_omegas]  # (b,option_dim)
    max_omegas_onehot_extended = \
        np.expand_dims(max_omegas_onehot, axis=1)  # (b,1,option_dim)

    q_omega_max = \
        tf.reduce_sum(agents_q_omega * max_omegas_onehot_extended, axis=-1)  # (b,n)

    return q_omega_max


if __name__ == '__main__':
    is_debug = False  # True for debug

    learn(num_workers=4, is_debug=is_debug)  # default num_workers=6

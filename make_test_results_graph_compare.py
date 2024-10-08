import os
import matplotlib.pyplot as plt
import json
from pathlib import Path


def draw_win_ratio_graphs(agent_type, savedir, x,
                          num_red_win_lists, num_blue_win_lists, num_no_contest_lists):
    plt.figure(figsize=(16, 12))
    idx = 0
    for num_red_win_list, num_blue_win_list, num_no_contest_list in \
            zip(num_red_win_lists, num_blue_win_lists, num_no_contest_lists):
        if idx == 0:
            pre_label = 'POMDP-1: '
            linestyle = 'solid'
        elif idx == 1:
            pre_label = 'POMDP-4: '
            linestyle = 'dashed'
        elif idx == 2:
            pre_label = 'IR: '
            linestyle = 'dotted'
        idx += 1

        plt.plot(x, num_red_win_list, color='r', marker='o', label=pre_label + 'red win',
                 linestyle=linestyle)
        plt.plot(x, num_blue_win_list, color='b', marker='o', label=pre_label + 'blue win',
                 linestyle=linestyle)
        plt.plot(x, num_no_contest_list, color='g', marker='s', label=pre_label + 'no contest',
                 linestyle=linestyle)

    plt.title('red win / blue win / no contest ratio, when increase ' + agent_type, fontsize="14")
    plt.xlabel('num (bin range center) of ' + agent_type, fontsize="14")
    plt.ylabel('win ratio', fontsize="14")
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    plt.legend(fontsize="14")
    plt.grid()

    savename = 'win_ratio_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()


def draw_alive_ratio_graphs(agent_type, savedir, x,
                            num_alive_reds_ratio_lists, num_alive_blues_ratio_lists):
    plt.figure(figsize=(16, 12))
    idx = 0
    for num_alive_reds_ratio_list, num_alive_blues_ratio_list in \
            zip(num_alive_reds_ratio_lists, num_alive_blues_ratio_lists):
        if idx == 0:
            pre_label = 'POMDP-1: '
            linestyle = 'solid'
        elif idx == 1:
            pre_label = 'POMDP-4: '
            linestyle = 'dashed'
        elif idx == 2:
            pre_label = 'IR: '
            linestyle = 'dotted'
        idx += 1

        plt.plot(x, num_alive_reds_ratio_list, color='r', marker='o',
                 label=pre_label + 'alive red', linestyle=linestyle)
        plt.plot(x, num_alive_blues_ratio_list, color='b', marker='o',
                 label=pre_label + 'alive blue', linestyle=linestyle)

    plt.title('num survive agents ratio, when increase ' + agent_type, fontsize="14")
    plt.xlabel('num (bin range center) of ' + agent_type, fontsize="14")
    plt.ylabel('alive agents ratio', fontsize="14")
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    plt.legend(fontsize="14")
    plt.grid()
    # plt.yscale('log')

    savename = 'alive_agents_ratio_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()


def draw_remaining_force_graph(agent_type, savedir, x,
                               remaining_red_effective_force_ratio_lists,
                               remaining_blue_effective_force_ratio_lists):
    plt.figure(figsize=(16, 12))
    idx = 0
    for remaining_red_effective_force_ratio_list, remaining_blue_effective_force_ratio_list in \
            zip(remaining_red_effective_force_ratio_lists,
                remaining_blue_effective_force_ratio_lists):
        if idx == 0:
            pre_label = 'POMDP-1: '
            linestyle = 'solid'
        elif idx == 1:
            pre_label = 'POMDP-4: '
            linestyle = 'dashed'
        elif idx == 2:
            pre_label = 'IR: '
            linestyle = 'dotted'
        idx += 1

        plt.plot(x, remaining_red_effective_force_ratio_list,
                 color='r', marker='o', label=pre_label + 'reds force', linestyle=linestyle)
        plt.plot(x, remaining_blue_effective_force_ratio_list,
                 color='b', marker='o', label=pre_label + 'blues force', linestyle=linestyle)

    plt.title('remaining force ratio, when increase ' + agent_type, fontsize="14")
    plt.xlabel('num (bin range center) of ' + agent_type, fontsize="14")
    plt.ylabel('total remaining force ratio', fontsize="14")
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    plt.legend(fontsize="14")
    plt.grid()
    # plt.yscale('log')

    savename = 'remaining_force_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()


def draw_return_len_graph(agent_type, savedir, x, episode_rewards_lists, episode_team_return_lists,
                          episode_lens_lists):
    plt.figure(figsize=(16, 12))
    idx = 0
    for episode_rewards_list, episode_team_return_list, episode_lens_list in \
            zip(episode_rewards_lists, episode_team_return_lists, episode_lens_lists):
        if idx == 0:
            pre_label = 'POMDP-1: '
            linestyle = 'solid'
        elif idx == 1:
            pre_label = 'POMDP-4: '
            linestyle = 'dashed'
        elif idx == 2:
            pre_label = 'IR: '
            linestyle = 'dotted'
        idx += 1

        plt.plot(x, episode_rewards_list, color='r', marker='o',
                 label=pre_label + 'sum of ind. return', linestyle=linestyle)
        plt.plot(x, episode_team_return_list, color='b', marker='o',
                 label=pre_label + 'team_return', linestyle=linestyle)
        plt.plot(x, episode_lens_list, color='g', marker='s', label=pre_label + 'episode len',
                 linestyle=linestyle)

    plt.title('returns and lens of episodes, when increase ' + agent_type, fontsize="14")
    plt.xlabel('num (bin range center) of ' + agent_type, fontsize="14")
    plt.ylabel('returns and length', fontsize="14")
    plt.minorticks_on()
    plt.legend(fontsize="14")
    plt.grid()

    savename = 'returns_length_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()


def make_test_results_graph_of_increase_number(agent_type, parent_dir, file_dir):
    num_red_win_list = []
    num_blue_win_list = []
    num_no_contest_list = []

    num_alive_reds_ratio_list = []
    num_alive_blues_ratio_list = []

    remaining_red_effective_force_ratio_list = []
    remaining_blue_effective_force_ratio_list = []

    episode_rewards_list = []
    episode_team_return_list = []
    episode_lens_list = []

    for file_name in file_dir:
        child_dir = agent_type + '=' + file_name + '/result.json'

        with open(parent_dir + child_dir, 'r') as f:
            json_data = json.load(f)

            num_red_win_list.append(json_data['num_red_win'] / 1000)
            num_blue_win_list.append(json_data['num_blue_win'] / 1000)
            num_no_contest_list.append(json_data['no_contest'] / 1000)

            num_alive_reds_ratio_list.append(json_data['num_alive_reds_ratio'])
            num_alive_blues_ratio_list.append(json_data['num_alive_blues_ratio'])

            remaining_red_effective_force_ratio_list. \
                append(json_data['remaining_red_effective_force_ratio'])
            remaining_blue_effective_force_ratio_list. \
                append(json_data['remaining_blue_effective_force_ratio'])

            episode_rewards_list.append(json_data['episode_rewards'])
            episode_team_return_list.append(json_data['episode_team_return'])
            episode_lens_list.append(json_data['episode_lens'])

    return [num_red_win_list, num_blue_win_list, num_no_contest_list, \
            num_alive_reds_ratio_list, num_alive_blues_ratio_list, \
            remaining_red_effective_force_ratio_list, remaining_blue_effective_force_ratio_list, \
            episode_rewards_list, episode_team_return_list, episode_lens_list]


def main():
    """ Select one of followings. """
    # agent_type = 'platoons'
    # agent_type = 'companies'
    agent_type = 'blue_platoons'

    if agent_type == 'platoons' or agent_type == 'blue_platoons' or agent_type == 'red_platoons':
        file_dir = ['(3,10)', '(11,20)', '(21,30)', '(31,40)', '(41,50)']
    elif agent_type == 'companies':
        file_dir = ['(6,10)', '(11,20)', '(21,30)', '(31,40)', '(41,50)']
    else:
        raise NotImplementedError()

    if agent_type == 'platoons' or agent_type == 'blue_platoons' or agent_type == 'red_platoons':
        x = [6.5, 15.5, 25.5, 35.5, 45.5]
    elif agent_type == 'companies':
        x = [3.5, 8.0, 15.5, 25.5, 35.5, 45.5]
    else:
        NotImplementedError()

    num_red_win_lists = []
    num_blue_win_lists = []
    num_no_contest_lists = []
    num_alive_reds_ratio_lists = []
    num_alive_blues_ratio_lists = []
    remaining_red_effective_force_ratio_lists = []
    remaining_blue_effective_force_ratio_lists = []
    episode_rewards_lists = []
    episode_team_return_lists = []
    episode_lens_lists = []

    """ team_reward, global_n_frames=1 """
    parent_dir = '13_MTC_SAC_DecPOMDP/trial_0_global_n_frames=1/robustness_test/robustness_blues/'

    results = make_test_results_graph_of_increase_number(agent_type, parent_dir, file_dir)
    num_red_win_lists.append(results[0])
    num_blue_win_lists.append(results[1])
    num_no_contest_lists.append(results[2])
    num_alive_reds_ratio_lists.append(results[3])
    num_alive_blues_ratio_lists.append(results[4])
    remaining_red_effective_force_ratio_lists.append(results[5])
    remaining_blue_effective_force_ratio_lists.append(results[6])
    episode_rewards_lists.append(results[7])
    episode_team_return_lists.append(results[8])
    episode_lens_lists.append(results[9])

    """ team_reward, global_n_frames=4 """
    parent_dir = '13_MTC_SAC_DecPOMDP/trial_1_global_n_frames=4/robustness_test/robustness_blues/'

    results = make_test_results_graph_of_increase_number(agent_type, parent_dir, file_dir)
    num_red_win_lists.append(results[0])
    num_blue_win_lists.append(results[1])
    num_no_contest_lists.append(results[2])
    num_alive_reds_ratio_lists.append(results[3])
    num_alive_blues_ratio_lists.append(results[4])
    remaining_red_effective_force_ratio_lists.append(results[5])
    remaining_blue_effective_force_ratio_lists.append(results[6])
    episode_rewards_lists.append(results[7])
    episode_team_return_lists.append(results[8])
    episode_lens_lists.append(results[9])

    """ individual_reward, global_n_frames=4 """
    parent_dir = '14_MTC_SAC_DecPOMDP_2/trial_0/robustness_test/robustness_blues/'

    savedir = Path(__file__).parent / 'robustness_blues_plot'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    results = make_test_results_graph_of_increase_number(agent_type, parent_dir, file_dir)
    num_red_win_lists.append(results[0])
    num_blue_win_lists.append(results[1])
    num_no_contest_lists.append(results[2])
    num_alive_reds_ratio_lists.append(results[3])
    num_alive_blues_ratio_lists.append(results[4])
    remaining_red_effective_force_ratio_lists.append(results[5])
    remaining_blue_effective_force_ratio_lists.append(results[6])
    episode_rewards_lists.append(results[7])
    episode_team_return_lists.append(results[8])
    episode_lens_lists.append(results[9])

    """ draw graphs """
    draw_win_ratio_graphs(agent_type, savedir, x,
                          num_red_win_lists, num_blue_win_lists, num_no_contest_lists)

    draw_alive_ratio_graphs(agent_type, savedir, x,
                            num_alive_reds_ratio_lists, num_alive_blues_ratio_lists)

    draw_remaining_force_graph(agent_type, savedir, x,
                               remaining_red_effective_force_ratio_lists,
                               remaining_blue_effective_force_ratio_lists)

    draw_return_len_graph(agent_type, savedir, x, episode_rewards_lists,
                          episode_team_return_lists, episode_lens_lists)


if __name__ == '__main__':
    main()

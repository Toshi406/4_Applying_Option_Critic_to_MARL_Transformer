import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np


def main():
    mode = 'learning'
    # mode = 'pre-training'
    # mode = 'fine-tuning'

    filelist = [
        "16_MTC_SAC_SelfPlay/test_trial_0/history",
        "16_MTC_SAC_SelfPlay/test_trial_1/history",
        "17_MTC_SAC_SelfPlay_NewReward/trial/history",
    ]

    filetypes = [
        '/run-.-tag-mean_num_alive_red_ratio.csv',
        '/run-.-tag-mean_remaining_red_effective_force_ratio.csv',
    ]

    legend_list_1 = ['POMDP-1: ', 'POMDP-4: ', 'IR: ']
    legend_list_2 = ['survive red agents', 'remaining red forces']

    colorlist = ['b', 'r', 'g', 'm']
    linelist = ['-', '--', ':', '-.']

    window = 10
    plt.figure(figsize=(16, 12))

    for f, c, l1 in zip(filelist, colorlist, legend_list_1):
        for filetype, l2, line in zip(filetypes, legend_list_2, linelist):

            ff = f + filetype
            csv_path = Path(__file__).parent / ff

            csv_df = pd.read_csv(csv_path)

            wall_time = csv_df[csv_df.columns[0]]
            step = csv_df[csv_df.columns[1]]
            values = csv_df[csv_df.columns[2]]

            averaged_wall_time = []
            averaged_step = []
            averaged_values = []

            for idx in range(len(values) - window + 1):
                averaged_wall_time.append(
                    np.mean(wall_time[idx:idx + window])
                )

                averaged_step.append(
                    np.mean(step[idx:idx + window])
                )

                averaged_values.append(
                    np.mean(values[idx:idx + window])
                )

            averaged_step = np.array(averaged_step)
            averaged_values = np.array(averaged_values)
            averaged_wall_time = np.array(averaged_wall_time)

            plt.plot((averaged_wall_time-averaged_wall_time[0]) / 3600, averaged_values,
                     linestyle=line, color=c, alpha=0.7, linewidth=2, label=l1 + l2)

    # plt.yscale('log')
    plt.title(f'Moving Average of remaining reds & forces, window={window}', fontsize="14")
    plt.grid(which="both")
    plt.xlabel(mode + ' hours', fontsize="14")
    plt.ylabel('remaining ratio', fontsize="14")
    plt.minorticks_on()
    plt.legend(fontsize="14")

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = filetype.replace('/run-.-tag-', '')
    savename = savename.replace('.csv', '')
    plt.savefig(str(savedir) + '/reds_time' + '.png', dpi=500)

    plt.show()


if __name__ == '__main__':
    main()

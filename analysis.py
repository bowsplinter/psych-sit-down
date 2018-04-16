control_group_time = [1148,1084,1480,1474,1462,1455,1435,1385,1380,1363]
control_group_score = [3,4,1,1,0,5,2,1,2,2]

experiment_group_time = [600,790,1020,713,770,1279,668,932,1102,806,735]
experiment_group_score = [1,4,0,2,2,2,1,1,1,0,2]

import math
import statistics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def descriptive_stats():
    # mean
    control_group_time_mean = statistics.mean(control_group_time)
    control_group_score_mean = statistics.mean(control_group_score)
    experiment_group_time_mean = statistics.mean(experiment_group_time)
    experiment_group_score_mean = statistics.mean(experiment_group_score)

    # median
    control_group_time_median = statistics.median(control_group_time)
    control_group_score_median = statistics.median(control_group_score)
    experiment_group_time_median = statistics.median(experiment_group_time)
    experiment_group_score_median = statistics.median(experiment_group_score)

    # compare mean and median across different
    plt.figure(1)
    bar_width = 0.3
    plt.subplot(121)
    plt.title("Time taken across Groups")
    bars1 = [control_group_time_mean, experiment_group_time_mean]
    bars2 = [control_group_time_median, experiment_group_time_median]
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    plt.bar(r1, bars1, width = bar_width, color = 'blue', edgecolor = 'black', capsize=7, label='mean')
    plt.bar(r2, bars2, width = bar_width, color = 'cyan', edgecolor = 'black', capsize=7, label='median')
    plt.xticks([r + bar_width for r in range(len(bars1))], ['Control Group', 'Experiment'])
    plt.ylabel('Time taken (s)')
    plt.legend()

    plt.subplot(122)
    plt.title("Scores across Groups")
    bars1 = [control_group_score_mean, experiment_group_score_mean]
    bars2 = [control_group_score_median, experiment_group_score_median]
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    plt.bar(r1, bars1, width = bar_width, color = 'blue', edgecolor = 'black', capsize=7, label='mean')
    plt.bar(r2, bars2, width = bar_width, color = 'cyan', edgecolor = 'black', capsize=7, label='median')
    plt.xticks([r + bar_width for r in range(len(bars1))], ['Control Group', 'Experiment'])
    plt.ylabel('Score achieved (out of 5)')
    plt.legend()

    plt.subplots_adjust(wspace=0.35)
    plt.show()

def check_normal_distribution():
    # Check data is normally distributed
    plt.figure(2)

    plt.subplot(221)
    plt.title("Control Group Times")
    sns.distplot(control_group_time)
    plt.subplot(222)
    plt.title("Control Group Scores")
    sns.distplot(control_group_score)
    plt.subplot(223)
    plt.title("Experiment Group Times")
    sns.distplot(experiment_group_time)
    plt.subplot(224)
    plt.title("Experiment Group Scores")
    sns.distplot(experiment_group_score)

    plt.subplots_adjust(hspace=0.35)
    plt.show()

if __name__ == '__main__':
    check_normal_distribution()
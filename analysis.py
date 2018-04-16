control_group_time = [1148,1084,1480,1474,1462,1455,1435,1385,1380,1363]
control_group_score = [3,4,1,1,0,5,2,1,2,2]

experiment_group_time = [600,790,1020,713,770,1279,668,932,1102,806,735]
experiment_group_score = [1,4,0,2,2,2,1,1,1,0,2]

import math
import statistics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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
    plt.bar(r1, bars1, width = bar_width, color = '#F15854', edgecolor = 'black', capsize=7, label='mean')
    plt.bar(r2, bars2, width = bar_width, color = '#FAA43A', edgecolor = 'black', capsize=7, label='median')
    plt.xticks([r + bar_width for r in range(len(bars1))], ['Control Group', 'Experiment'])
    plt.ylabel('Time taken (s)')
    plt.legend()

    plt.subplot(122)
    plt.title("Scores across Groups")
    bars1 = [control_group_score_mean, experiment_group_score_mean]
    bars2 = [control_group_score_median, experiment_group_score_median]
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    plt.bar(r1, bars1, width = bar_width, color = '#F15854', edgecolor = 'black', capsize=7, label='mean')
    plt.bar(r2, bars2, width = bar_width, color = '#FAA43A', edgecolor = 'black', capsize=7, label='median')
    plt.xticks([r + bar_width for r in range(len(bars1))], ['Control Group', 'Experiment'])
    plt.ylabel('Score achieved (out of 5)')
    plt.legend()

    plt.subplots_adjust(wspace=0.35)
    plt.show()

def check_normal_distribution():
    # Check data is normally distributed
    plt.figure(2)
    plt.subplot(221)
    plt.tick_params(
        axis='both',         # changes apply to both axis
        which='both',       # both major and minor ticks are affected
        bottom='off',       # ticks along the bottom edge are off
        left='off',         # ticks along the left edge are off
        labelleft='off',    # labels along the left edge are off
        labelbottom='off')  # labels along the bottom edge are off
    plt.title("Control Group Times")
    sns.distplot(control_group_time)
    plt.subplot(222)
    plt.tick_params(
        axis='both',         # changes apply to both axis
        which='both',       # both major and minor ticks are affected
        bottom='off',       # ticks along the bottom edge are off
        left='off',         # ticks along the left edge are off
        labelleft='off',    # labels along the left edge are off
        labelbottom='off')  # labels along the bottom edge are off
    plt.title("Control Group Scores")
    sns.distplot(control_group_score, bins=5)
    plt.subplot(223)
    plt.tick_params(
        axis='both',         # changes apply to both axis
        which='both',       # both major and minor ticks are affected
        bottom='off',       # ticks along the bottom edge are off
        left='off',         # ticks along the left edge are off
        labelleft='off',    # labels along the left edge are off
        labelbottom='off')  # labels along the bottom edge are off
    plt.title("Experiment Group Times")
    sns.distplot(experiment_group_time)
    plt.subplot(224)
    plt.tick_params(
        axis='both',         # changes apply to both axis
        which='both',       # both major and minor ticks are affected
        bottom='off',       # ticks along the bottom edge are off
        left='off',         # ticks along the left edge are off
        labelleft='off',    # labels along the left edge are off
        labelbottom='off')  # labels along the bottom edge are off
    plt.title("Experiment Group Scores")
    sns.distplot(experiment_group_score, bins=5)
    plt.subplots_adjust(hspace=0.35)

    plt.show()

def anderson_darling_test(x):
    # https://en.wikipedia.org/wiki/Anderson–Darling_test
    x = np.array(x, dtype='float')
    mean = statistics.mean(x)
    variance = statistics.variance(x)
    normal_x = (x-mean)/math.sqrt(variance)
    normal_cdf_x = [stats.norm.cdf(i) for i in normal_x]
    a_squared_temp = 0
    num_x = len(x)
    for i, phi in enumerate(normal_cdf_x):
        i = i+1
        a_squared_temp += (2*i - 1)*np.log(phi) + (2*(num_x-i) + 1)*np.log(1-phi)
    a_squared_temp /= num_x
    a_squared = - num_x - a_squared_temp
    a_squared *= (1 + 4.0/num_x + 25.0/math.pow(num_x,2))


if __name__ == '__main__':
    check_normal_distribution()
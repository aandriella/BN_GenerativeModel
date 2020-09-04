import numpy as np
import matplotlib.pyplot as plt


def plot2D_game_performance(save_path, n_episodes, *y):
    # The position of the bars on the x-axis
    barWidth = 0.35
    r = np.arange(n_episodes)  # the x locations for the groups
    # Get values from the group and categories
    x = [i for i in range(n_episodes)]
    correct = list(map(lambda x:x[0], y[0]))
    wrong = list(map(lambda x:x[1], y[0]))
    timeout = list(map(lambda x:x[2], y[0]))
    max_attempt = list(map(lambda x:x[3], y[0]))

    # plot bars
    plt.figure(figsize=(10, 7))
    plt.bar(r, correct,  edgecolor='white', width=barWidth, label="correct")
    plt.bar(r, wrong, bottom=np.array(correct), edgecolor='white', width=barWidth, label='wrong')
    plt.bar(r, timeout, bottom=np.array(correct) + np.array(wrong), edgecolor='white',
            width=barWidth, label='timeout')
    plt.bar(r, max_attempt, bottom=np.array(correct) + np.array(wrong) + np.array(timeout), edgecolor='white',
            width=barWidth, label='max_attempt')

    plt.legend()
    # Custom X axis
    plt.xticks(r, x, fontweight='bold')
    plt.ylabel("performance")
    plt.savefig(save_path)
    plt.show()


def plot2D_assistance(save_path, n_episodes, *y):
    # The position of the bars on the x-axis
    barWidth = 0.35
    r = np.arange(n_episodes)  # the x locations for the groups
    # Get values from the group and categories
    x = [i for i in range(n_episodes)]

    lev_0_no_feed = list(map(lambda x:x[0], y[0]))
    lev_1_no_feed = list(map(lambda x:x[1], y[0]))
    lev_2_no_feed = list(map(lambda x:x[2], y[0]))
    lev_3_no_feed = list(map(lambda x:x[3], y[0]))
    lev_4_no_feed = list(map(lambda x:x[4], y[0]))
    lev_0_with_feed = list(map(lambda x:x[5], y[0]))
    lev_1_with_feed = list(map(lambda x:x[6], y[0]))
    lev_2_with_feed = list(map(lambda x:x[7], y[0]))
    lev_3_with_feed = list(map(lambda x:x[8], y[0]))
    lev_4_with_feed = list(map(lambda x:x[9], y[0]))

    # plot bars
    plt.figure(figsize=(10, 7))
    plt.bar(r, lev_0_no_feed, edgecolor='white', width=barWidth, label="lev_0_no_feed")
    plt.bar(r, lev_1_no_feed, bottom=np.array(lev_0_no_feed), edgecolor='white', width=barWidth, label='lev_1_no_feed')
    plt.bar(r, lev_2_no_feed, bottom=np.array(lev_0_no_feed) + np.array(lev_1_no_feed), edgecolor='white',
            width=barWidth, label='lev_2_no_feed')
    plt.bar(r, lev_3_no_feed, bottom=np.array(lev_0_no_feed) + np.array(lev_1_no_feed)+ np.array(lev_2_no_feed), edgecolor='white',
            width=barWidth, label='lev_3_no_feed')
    plt.bar(r, lev_4_no_feed, bottom=np.array(lev_0_no_feed) + np.array(lev_1_no_feed)+ np.array(lev_2_no_feed)+ np.array(lev_3_no_feed), edgecolor='white',
            width=barWidth, label='lev_4_no_feed')
    plt.bar(r, lev_0_with_feed, bottom=np.array(lev_0_no_feed) + np.array(lev_1_no_feed)+ np.array(lev_2_no_feed)+ np.array(lev_3_no_feed)+ np.array(lev_4_no_feed), edgecolor='white',
            width=barWidth, label='lev_0_with_feed')
    plt.bar(r, lev_1_with_feed, bottom=np.array(lev_0_no_feed) + np.array(lev_1_no_feed)+ np.array(lev_2_no_feed)+ np.array(lev_3_no_feed)+ np.array(lev_4_no_feed)+ np.array(lev_0_with_feed), edgecolor='white',
            width=barWidth, label='lev_1_with_feed')
    plt.bar(r, lev_2_with_feed, bottom=np.array(lev_0_no_feed) + np.array(lev_1_no_feed)+ np.array(lev_2_no_feed)+ np.array(lev_3_no_feed)+ np.array(lev_4_no_feed)+ np.array(lev_0_with_feed)+ np.array(lev_1_with_feed), edgecolor='white',
            width=barWidth, label='lev_2_with_feed')
    plt.bar(r, lev_3_with_feed, bottom=np.array(lev_0_no_feed) + np.array(lev_1_no_feed)+ np.array(lev_2_no_feed)+ np.array(lev_3_no_feed)+ np.array(lev_4_no_feed)+ np.array(lev_0_with_feed)+ np.array(lev_1_with_feed)+ np.array(lev_2_with_feed), edgecolor='white',
            width=barWidth, label='lev_3_with_feed')
    plt.bar(r, lev_4_with_feed, bottom=np.array(lev_0_no_feed) + np.array(lev_1_no_feed)+ np.array(lev_2_no_feed)+ np.array(lev_3_no_feed)+ np.array(lev_4_no_feed)+ np.array(lev_0_with_feed)+ np.array(lev_1_with_feed)+ np.array(lev_2_with_feed)+ np.array(lev_3_with_feed), edgecolor='white',
            width=barWidth, label='lev_4_with_feed')

    plt.legend()
    # Custom X axis
    plt.xticks(r, x, fontweight='bold')
    plt.ylabel("performance")
    plt.savefig(save_path)
    plt.show()
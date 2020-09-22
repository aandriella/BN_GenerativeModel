import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot2D_game_performance(save_path, n_episodes, *y):
    # The position of the bars on the x-axis
    barWidth = 0.35
    r = np.arange(n_episodes)[1::10]  # the x locations for the groups
    # Get values from the group and categories
    x = [i for i in range(n_episodes)][1::10]
    correct = list(map(lambda x:x[0], y[0]))[1::10]
    wrong = list(map(lambda x:x[1], y[0]))[1::10]
    timeout = list(map(lambda x:x[2], y[0]))[1::10]
    max_attempt = list(map(lambda x:x[3], y[0]))[1::10]

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
    r = np.arange(n_episodes)[1::10]  # the x locations for the groups
    # Get values from the group and categories
    x = [i for i in range(n_episodes)][1::10]

    lev_0 = list(map(lambda x:x[0], y[0]))[1::10]
    lev_1 = list(map(lambda x:x[1], y[0]))[1::10]
    lev_2 = list(map(lambda x:x[2], y[0]))[1::10]
    lev_3 = list(map(lambda x:x[3], y[0]))[1::10]
    lev_4 = list(map(lambda x:x[4], y[0]))[1::10]

    # plot bars
    plt.figure(figsize=(10, 7))
    plt.bar(r, lev_0, edgecolor='white', width=barWidth, label="lev_0")
    plt.bar(r, lev_1, bottom=np.array(lev_0), edgecolor='white', width=barWidth, label='lev_1')
    plt.bar(r, lev_2, bottom=np.array(lev_0) + np.array(lev_1), edgecolor='white',
            width=barWidth, label='lev_2')
    plt.bar(r, lev_3, bottom=np.array(lev_0) + np.array(lev_1)+ np.array(lev_2), edgecolor='white',
            width=barWidth, label='lev_3')
    plt.bar(r, lev_4, bottom=np.array(lev_0) + np.array(lev_1)+ np.array(lev_2)+ np.array(lev_3), edgecolor='white',
            width=barWidth, label='lev_4')


    plt.legend()
    # Custom X axis
    plt.xticks(r, x, fontweight='bold')
    plt.ylabel("assistance")
    plt.savefig(save_path)
    plt.show()

def plot2D_feedback(save_path, n_episodes, *y):
    # The position of the bars on the x-axis
    barWidth = 0.35
    r = np.arange(n_episodes)[1::10]  # the x locations for the groups
    # Get values from the group and categories
    x = [i for i in range(n_episodes)][1::10]

    feedback_no = list(map(lambda x:x[0], y[0]))[1::10]
    feedback_yes = list(map(lambda x:x[1], y[0]))[1::10]

    # plot bars
    plt.figure(figsize=(10, 7))
    plt.bar(r, feedback_no, edgecolor='white', width=barWidth, label="feedback_no")
    plt.bar(r, feedback_yes, bottom=np.array(feedback_no), edgecolor='white', width=barWidth, label='feedback_yes')
    plt.legend()
    # Custom X axis
    plt.xticks(r, x, fontweight='bold')
    plt.ylabel("feedback")
    plt.savefig(save_path)
    plt.show()

def read_user_statistics_from_pickle(file_name):
    # Getting back the objects:
    with open(file_name, 'rb') as handle:
        bn_dict_vars = pickle.load(handle)
    return bn_dict_vars
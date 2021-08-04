import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot2D_game_performance(save_path, n_episodes, scaling_factor=1, *y):
    # The position of the bars on the x-axis
    barWidth = 0.35
    r = np.arange(n_episodes)  # the x locations for the groups
    # Get values from the group and categories
    x = [i for i in range(1, n_episodes+1)]
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

    plt.legend(loc="upper right")
    # Custom X axis
    plt.xticks(r, x, fontweight='bold')
    plt.ylabel("sim patient performance")
    plt.xlabel("epoch")
    plt.savefig(save_path)


def plot2D_assistance(save_path, n_episodes, scaling_factor=1, *y):
    # The position of the bars on the x-axis
    barWidth = 0.35
    r = np.arange(n_episodes+1)
    # the x locations for the groups
    # Get values from the group and categories
    x = [i for i in range(1, n_episodes+2)]

    lev_0 = list(map(lambda x:x[0], y[0]))
    lev_1 = list(map(lambda x:x[1], y[0]))
    lev_2 = list(map(lambda x:x[2], y[0]))
    lev_3 = list(map(lambda x:x[3], y[0]))
    lev_4 = list(map(lambda x:x[4], y[0]))
    lev_5 = list(map(lambda x:x[5], y[0]))
    lev_6 = list(map(lambda x:x[6], y[0]))

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
    plt.bar(r, lev_5, bottom=np.array(lev_0) + np.array(lev_1) + np.array(lev_2) + np.array(lev_3)+ np.array(lev_4), edgecolor='white',
            width=barWidth, label='lev_5')
    plt.bar(r, lev_6, bottom=np.array(lev_0) + np.array(lev_1) + np.array(lev_2) + np.array(lev_3) + np.array(lev_4)+np.array(lev_5),
            edgecolor='white',
            width=barWidth, label='lev_6')

    plt.legend(loc="upper right")
    # Custom X axis
    plt.xticks(r, x, fontweight='bold')
    plt.ylabel("Levels of assistance")
    plt.xlabel("Epoch")
    plt.savefig(save_path)

def plot2D_feedback(save_path, n_episodes, scaling_factor=1, *y):
    # The position of the bars on the x-axis
    barWidth = 0.35
    r = np.arange(n_episodes)[1::scaling_factor+1]  # the x locations for the groups
    # Get values from the group and categories
    x = [i for i in range(n_episodes)][1::scaling_factor+1]

    feedback_no = list(map(lambda x:x[0], y[0]))[0::scaling_factor]
    feedback_yes = list(map(lambda x:x[1], y[0]))[0::scaling_factor]

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
import bnlearn
import numpy as np
import enum
import random
import matplotlib.pyplot as plt

#define constants
class User_Action(enum.Enum):
    correct = 0
    wrong = 1
    timeout = 2
    name = "user_action"
    counter = 3
class Reactivity(enum.Enum):
    slow = 0
    medium = 1
    fast = 2
    name = "reactivity"
    counter = 3
class Memory(enum.Enum):
    low = 0
    medium = 1
    high = 2
    name = "memory"
    counter = 3
class Robot_Assistance(enum.Enum):
    lev_0 = 0
    lev_1 = 1
    lev_2 = 2
    lev_3 = 3
    lev_4 = 4
    name = "robot_assistance"
    counter = 5
class Robot_Feedback(enum.Enum):
    yes = 0
    false = 1
    name = "robot_feedback"
    counter = 2
class Game_State(enum.Enum):
    beg = 0
    middle = 1
    end = 2
    name = "game_state"
    counter = 3
class Attempt(enum.Enum):
    at_1 = 0
    at_2 = 1
    at_3 = 2
    at_4 = 3
    name = "attempt"
    counter = 4



def plot2D(save_path, n_episodes, *y):
    # The position of the bars on the x-axis
    barWidth = 0.35
    r = np.arange(n_episodes)  # the x locations for the groups

    # Get values from the group and categories
    x = [i for i in range(n_episodes)]
    correct = y[0][0]
    wrong = y[0][1]
    timeout = y[0][2]
    # add colors
    #colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']

    # plot bars
    plt.figure(figsize=(10, 7))
    plt.bar(r, correct,  edgecolor='white', width=barWidth, label="correct")
    plt.bar(r, wrong, bottom=np.array(correct), edgecolor='white', width=barWidth, label='wrong')
    plt.bar(r, timeout, bottom=np.array(correct) + np.array(wrong), edgecolor='white',
            width=barWidth, label='timeout')
    plt.legend()
    # Custom X axis
    plt.xticks(r, x, fontweight='bold')
    plt.ylabel("performance")
    plt.savefig(save_path)
    plt.show()

def compute_prob(cpds_table):
    '''
    Given the counters generate the probability distributions
    Args:
        cpds_table: with counters
    Return:
         the probs for the cpds table
    '''
    for val in range(len(cpds_table)):
            cpds_table[val] = list(map(lambda x: x / (sum(cpds_table[val])+0.00001), cpds_table[val]))
    return cpds_table

def average_prob(ref_cpds_table, current_cpds_table):
    '''
    Args:
        ref_cpds_table: table from bnlearn
        current_cpds_table: table from interaction
    Return:
        avg from both tables
    '''
    res_cpds_table = ref_cpds_table.copy()
    for elem1 in range(len(ref_cpds_table)):
        for elem2 in range(len(ref_cpds_table[0])):
            res_cpds_table[elem1][elem2] = (ref_cpds_table[elem1][elem2]+current_cpds_table[elem1][elem2])/2
    return res_cpds_table

def generate_user_action(actions_prob):
    '''
    Select one of the actions according to the actions_prob
    Args:
        actions_prob: the result of the query to the BN
    Return:
        the id of the selected action
    '''
    action_id = 0
    correct_action = actions_prob[0]
    wrong_action = actions_prob[1]
    timeout = actions_prob[2]
    rnd_val = random.random()
    if rnd_val<=correct_action:
        action_id = 0
    elif rnd_val>correct_action \
        and rnd_val<correct_action+wrong_action:
        action_id = 1
    else:
        action_id = 2
    return action_id


def simulation(robot_assistance_vect, robot_feedback_vect, persona_cpds, memory, attention, reactivity, epochs=50, task_complexity=5, non_stochastic=False):
    #metrics we need in order to compute the afterwords the belief
    '''
    CPD 0: for each attempt 1 to 4 store the number of correct, wrong and timeout
    '''
    attempt_counter_per_action = [[0 for i in range(Attempt.counter.value)]  for j in range(User_Action.counter.value)]
    '''
    CPD 2: for each game_state 0 to 2 store the number of correct, wrong and timeout
    '''
    game_state_counter_per_action = [[0 for i in range(Game_State.counter.value)]  for j in range(User_Action.counter.value)]
    '''
    CPD 5: for each robot feedback store the number of correct, wrong and timeout
    '''
    robot_feedback_per_action = [[0 for i in range(Robot_Feedback.counter.value)] for j in range(User_Action.counter.value)]
    '''
    CPD 6: for each robot assistance store the number of pos and neg feedback
    '''
    robot_assistance_per_feedback = [[0 for i in range(Robot_Assistance.counter.value)] for j in range(Robot_Feedback.counter.value)]


    #output variables:
    n_correct_per_episode = [0]*epochs
    n_wrong_per_episode = [0]*epochs
    n_timeout_per_episode = [0]*epochs


    for e in range(epochs):
        '''Simulation framework'''
        #counters
        task_evolution = 0
        attempt_counter = 0
        iter_counter = 0
        correct_move_counter = 0
        wrong_move_counter = 0
        timeout_counter = 0

        while(task_evolution<=task_complexity):
            #these, if then else are necessary to classify the task game state into beg, mid, end
            if task_evolution>=0 and task_evolution<=2:
                game_state_counter = 0
            elif task_evolution>=3 and task_evolution<=4:
                game_state_counter = 1
            else:
                game_state_counter = 2
            #select robot assistance
            robot_assistance_action = random.randint(min(robot_assistance_vect), max(robot_assistance_vect))
            #select robot feedback
            robot_feedback_action = random.randint(min(robot_feedback_vect), max(robot_feedback_vect))

            print("robot_assistance {}, attempt {}, game {}, robot_feedback {}".format(robot_assistance_action, attempt_counter, game_state_counter, robot_feedback_action))
            query = bnlearn.inference.fit(persona_cpds, variables=['user_action'], evidence={'robot_assistance': robot_assistance_action,
                                                                                      'attempt': attempt_counter,
                                                                                      'game_state': game_state_counter,
                                                                                      'robot_feedback': robot_feedback_action,
                                                                                      'memory': memory,
                                                                                      'attention': attention,
                                                                                      'reactivity': reactivity
                                                                                      })
            #generate a random number and trigger one of the three possible action
            user_action = generate_user_action(query.values)#np.argmax(query.values, axis=0)

            #updates counters for plots
            robot_assistance_per_feedback[robot_feedback_action][robot_assistance_action] += 1
            attempt_counter_per_action[user_action][attempt_counter] += 1
            game_state_counter_per_action[user_action][game_state_counter] += 1
            robot_feedback_per_action[user_action][robot_feedback_action] += 1

            #updates counters for simulation
            iter_counter += 1
            if user_action == 0:
                attempt_counter = 0
                task_evolution += 1
                correct_move_counter += 1
            #if the user made a wrong move and still did not reach the maximum number of attempts
            elif user_action == 1 and attempt_counter<3:
                attempt_counter += 1
                wrong_move_counter += 1
            # if the user did not move any token and still did not reach the maximum number of attempts
            elif user_action == 2 and attempt_counter<3:
                attempt_counter += 1
                wrong_move_counter += 1
            # the robot or therapist makes the correct move on the patient behalf
            else:
                attempt_counter = 0
                task_evolution += 1
                timeout_counter += 1

        print("task_evolution {}, attempt_counter {}, timeout_counter {}".format(task_evolution, iter_counter, timeout_counter))

        print("robot_assistance_per_feedback {}".format(robot_assistance_per_feedback))
        print("attempt_counter_per_action {}".format(attempt_counter_per_action))
        print("game_state_counter_per_action {}".format(game_state_counter_per_action))
        print("robot_feedback_per_action {}".format(robot_feedback_per_action))
        print("iter {}, correct {}, wrong {}, timeout {}".format(iter_counter, correct_move_counter, wrong_move_counter, timeout_counter))

        print("correct_move {}, wrong_move {}, timeout {}".format(correct_move_counter, wrong_move_counter, timeout_counter))

        #transform counters into probabilities
        prob_over_attempt_per_action = compute_prob(attempt_counter_per_action)
        prob_over_game_per_action = compute_prob(game_state_counter_per_action)
        prob_over_feedback_per_action = compute_prob(robot_feedback_per_action)
        prob_over_assistance_per_feedback = compute_prob(robot_assistance_per_feedback)

        #average the probabilities obtained with the cpdf tables
        updated_prob_over_attempt_per_action = average_prob(np.transpose(persona_cpds['model'].cpds[0].values),
                                                        prob_over_attempt_per_action)
        updated_prob_over_game_per_action = average_prob(np.transpose(persona_cpds['model'].cpds[2].values),
                                                     prob_over_game_per_action)
        updated_prob_over_feedback_per_action = average_prob(np.transpose(persona_cpds['model'].cpds[6].values),
                                                         prob_over_feedback_per_action)
        updated_prob_over_assistance_per_feedback = average_prob(np.transpose(persona_cpds['model'].cpds[5].values),
                                                             prob_over_assistance_per_feedback)

        persona_cpds['model'].cpds[0].values = np.transpose(updated_prob_over_attempt_per_action)
        persona_cpds['model'].cpds[2].values = np.transpose(updated_prob_over_game_per_action)
        persona_cpds['model'].cpds[6].values = np.transpose(updated_prob_over_feedback_per_action)
        persona_cpds['model'].cpds[5].values = np.transpose(updated_prob_over_assistance_per_feedback)

        n_correct_per_episode[e] = correct_move_counter
        n_wrong_per_episode[e] = wrong_move_counter
        n_timeout_per_episode[e] = timeout_counter

    return n_correct_per_episode, n_wrong_per_episode, n_timeout_per_episode

robot_assistance = [i for i in range(Robot_Assistance.counter.value)]
robot_feedback = [i for i in range(Robot_Feedback.counter.value)]
epochs = 10
#initialise memory, attention and reactivity varibles
memory = 0; attention = 0; reactivity = 1;
#run a simulation
persona_cpds = bnlearn.import_DAG('persona_model.bif')
print("user_action -> attempt ", persona_cpds['model'].cpds[0].values)
print("user_action -> game_state ", persona_cpds['model'].cpds[2].values)
print("robot_feedback -> robot_assistance ", persona_cpds['model'].cpds[5].values)
print("user_action -> reactivity, memory ", persona_cpds['model'].cpds[6].values)

results = simulation(robot_assistance, robot_feedback, persona_cpds, memory, attention, reactivity, epochs=10, task_complexity=5, non_stochastic=False)
plot_path = "epoch_"+str(epochs)+"_memory_"+str(memory)+"_attention_"+str(attention)+"_reactivity_"+str(reactivity)+".jpg"
plot2D(plot_path, epochs, results)

#TODO
'''
- define a function that takes the state as input and return the user action and its reaction time
- evalute if the persona is wrong how long does it take for the simulator to detect that
- check percentages
'''
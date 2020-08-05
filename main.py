import bnlearn
import numpy as np
import enum
import random

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

model = bnlearn.import_DAG('persona_model_4.bif')
print("user_action -> attempt ", model['model'].cpds[0].values)
print("user_action -> game_state ", model['model'].cpds[2].values)
print("robot_feedback -> robot_assistance ", model['model'].cpds[5].values)
print("user_action -> reactivity, memory ", model['model'].cpds[6].values)

def compute_prob(cpds_table):
    for val in range(len(cpds_table)):
            cpds_table[val] = list(map(lambda x: x / (sum(cpds_table[val])+0.00001), cpds_table[val]))
    return cpds_table

def avg_prob(ref_cpds_table, current_cpds_table):
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


def simulation(robot_assistance_vect, robot_feedback_vect):
    #metrics we need in order to compute the afterwords the belief
    '''
    CPD 0: for each attempt 1 to 4 store the number of correct, wrong and timeout
    '''
    attempt_counter_per_action = [[0 for j in range(User_Action.counter.value)] for i in range(Attempt.counter.value)]
    '''
    CPD 2: for each game_state 0 to 2 store the number of correct, wrong and timeout
    '''
    game_state_counter_per_action = [[0 for j in range(User_Action.counter.value)] for i in range(Game_State.counter.value)]
    '''
    CPD 5: for each robot feedback store the number of correct, wrong and timeout
    '''
    robot_feedback_per_action = [[0 for j in range(User_Action.counter.value)] for i in range(Robot_Feedback.counter.value)]
    '''
    CPD 6: for each robot assistance store the number of pos and neg feedback
    '''
    robot_assistance_per_feedback = [[0 for j in range(Robot_Feedback.counter.value)] for i in range(Robot_Assistance.counter.value)]

    task_complexity = 5
    task_evolution = 0
    attempt_counter = 0
    game_state_counter = 0

    iter_counter = 0
    correct_move_counter = 0
    wrong_move_counter = 0
    timeout_counter = 0

    '''Simulation framework'''
    while(task_evolution<=task_complexity):
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
        query = bnlearn.inference.fit(model, variables=['user_action'], evidence={'robot_assistance': robot_assistance_action,
                                                                                  'attempt': attempt_counter,
                                                                                  'game_state': game_state_counter,
                                                                                  'robot_feedback': robot_feedback_action,
                                                                                  'memory': 0,
                                                                                  'attention': 0,
                                                                                  'reactivity': 0
                                                                                  })
        user_move_action = np.argmax(query.values, axis=0)

        robot_assistance_per_feedback[robot_assistance_action][robot_feedback_action] += 1
        attempt_counter_per_action[attempt_counter][user_move_action] += 1
        game_state_counter_per_action[game_state_counter][user_move_action] += 1
        robot_feedback_per_action[robot_feedback_action][user_move_action] += 1

        iter_counter += 1
        if user_move_action == 0:
            attempt_counter += 0
            task_evolution += 1
            correct_move_counter += 1
        elif user_move_action == 1 and attempt_counter<3:
            attempt_counter += 1
            wrong_move_counter += 1
        elif user_move_action == 2 and attempt_counter<3:
            attempt_counter += 1
            wrong_move_counter += 1
        else:
            attempt_counter += 0
            task_evolution += 1
            timeout_counter += 1

        print("correct {}, wrong {}, timeout {}".format(query.values[0],
                                                    query.values[1],
                                                    query.values[2]))


    print("robot_assistance_per_feedback {}".format(robot_assistance_per_feedback))
    print("attempt_counter_per_action {}".format(attempt_counter_per_action))
    print("game_state_counter_per_action {}".format(game_state_counter_per_action))
    print("robot_feedback_per_action {}".format(robot_feedback_per_action))
    print("iter {}, correct {}, wrong {}, timeout {}".format(iter_counter, correct_move_counter, wrong_move_counter, timeout_counter))

    return attempt_counter_per_action, game_state_counter_per_action, robot_assistance_per_feedback, robot_feedback_per_action


robot_assistance_vect = [0, 1, 2, 3, 4]
robot_feedback_vect = [0, 1]
attempt_counter_per_action, game_state_counter_per_action, \
robot_assistance_per_feedback, robot_feedback_per_action = simulation(robot_assistance_vect, robot_feedback_vect)

print("************BEFORE*************")
print(model['model'].cpds[0].values)
print(model['model'].cpds[2].values)
print(model['model'].cpds[5].values)
print(model['model'].cpds[6].values)

prob_over_attempt_per_action = compute_prob(attempt_counter_per_action)
prob_over_game_per_action  = compute_prob(game_state_counter_per_action)
prob_over_feedback_per_action = compute_prob(robot_feedback_per_action)
prob_over_assistance_per_feedback = compute_prob(robot_assistance_per_feedback)

print("************DURING*************")
print(prob_over_attempt_per_action)
print(prob_over_game_per_action)
print(prob_over_feedback_per_action)
print(prob_over_assistance_per_feedback)

res_prob_over_attempt_per_action = avg_prob(model['model'].cpds[0].values,
                                            prob_over_attempt_per_action)
res_prob_over_game_per_action = avg_prob(model['model'].cpds[2].values,
                                         prob_over_game_per_action)
res_prob_over_feedback_per_action = avg_prob(model['model'].cpds[6].values,
                                         prob_over_feedback_per_action)
res_prob_over_assistance_per_feedback = avg_prob(model['model'].cpds[5].values,
                                                 prob_over_assistance_per_feedback)


print("************AFTER*************")
print(res_prob_over_attempt_per_action)
print(res_prob_over_game_per_action)
print(res_prob_over_feedback_per_action)
print(res_prob_over_assistance_per_feedback)

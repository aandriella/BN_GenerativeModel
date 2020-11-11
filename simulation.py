import itertools
import os
import bnlearn
import numpy as np
import random
import copy
#import classes and modules
from bn_variables import Agent_Assistance, Agent_Feedback, User_Action, User_React_time, Game_State, Attempt
import bn_functions
import utils
from episode import Episode
import pandas as pd

def build_model_from_data(csv_filename, dag_filename, dag_model=None):
    print("/************************************************************/")
    print("Init model")
    DAG = bnlearn.import_DAG(dag_filename)
    df_caregiver = bnlearn.sampling(DAG, n=10000)

    print("/************************************************************/")
    print("real_user Model")
    DAG_real_user_no_cpd = bnlearn.import_DAG(dag_filename, CPD=False)
    df_real_user = pd.read_csv(csv_filename)
    DAG_real_user = bnlearn.parameter_learning.fit(DAG_real_user_no_cpd, df_real_user, methodtype='bayes')
    df_real_user = bnlearn.sampling(DAG_real_user, n=10000)
    print("/************************************************************/")
    print("Shared knowledge")
    DAG_shared_no_cpd = bnlearn.import_DAG(dag_filename, CPD=False)
    shared_knowledge = [df_real_user, df_caregiver]
    conc_shared_knowledge = pd.concat(shared_knowledge)
    DAG_shared = bnlearn.parameter_learning.fit(DAG_shared_no_cpd, conc_shared_knowledge)
    #df_conc_shared_knowledge = bn.sampling(DAG_shared, n=10000)
    return DAG_shared


def generate_agent_assistance(preferred_assistance, agent_behaviour, current_state, state_space, action_space):
    episode = Episode()
    game_state, attempt, prev_user_action = episode.state_from_index_to_point(state_space, current_state)
    robot_action = 0
    #agent_behaviour is a tuple first item is the feedback, second item is the robot assistance
    print(game_state,attempt, prev_user_action)
    if attempt == 1:
        robot_action = episode.state_from_point_to_index(action_space, (random.randint(0, 1), 0))
    elif attempt!=1 and prev_user_action == 0:
        if attempt == 2 and agent_behaviour == "challenge":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0,1),min(max(0, preferred_assistance-1), 5)))
            print("catt2")
        elif attempt == 2 and agent_behaviour == "help":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0, 1), min(max(0, preferred_assistance), 5)))
            print("hatt2")
        elif attempt == 3 and agent_behaviour == "challenge":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0,1),min(max(0, preferred_assistance-2), 5)))
            print("catt3")
        elif attempt == 3 and agent_behaviour == "help":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0, 1), min(max(0, preferred_assistance+1), 5)))
            print("hatt3")
        elif attempt == 4 and agent_behaviour == "challenge":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0,1),min(max(0, preferred_assistance-3), 5)))
            print("catt4")
        elif attempt == 4 and agent_behaviour == "help":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0, 1), min(max(0, preferred_assistance+2), 5)))
            print("hatt4")

    elif attempt!=1 and prev_user_action == -1:
        if attempt == 2 and agent_behaviour == "challenge":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0, 1), min(max(0, preferred_assistance+1), 5)))
            print("catt2")
        elif attempt == 2 and agent_behaviour == "help":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0, 1), min(max(0, preferred_assistance), 5)))
            print("hatt2")
        elif attempt == 3 and agent_behaviour == "challenge":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0, 1), min(max(0, preferred_assistance+2), 5)))
            print("catt3")
        elif attempt == 3 and agent_behaviour == "help":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0, 1), min(max(0, preferred_assistance-1), 5)))
            print("hatt3")
        elif attempt == 4 and agent_behaviour == "challenge":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0, 1), min(max(0, preferred_assistance+2), 5)))
            print("catt4")
        elif attempt == 4 and agent_behaviour == "help":
            robot_action = episode.state_from_point_to_index(action_space,
            (random.randint(0, 1), min(max(0, preferred_assistance-3), 5)))
            print("hatt4")

    agent_assistance = episode.state_from_index_to_point(action_space, robot_action)

    return agent_assistance




def compute_next_state(user_action, task_progress_counter, attempt_counter, correct_move_counter,
                       wrong_move_counter, timeout_counter, max_attempt_counter, max_attempt_per_object
                       ):
    '''
    The function computes given the current state and action of the user, the next state
    Args:
        user_action: -1a wrong, 0 timeout, 1a correct
        game_state_counter: beg, mid, end
        correct_move_counter:
        attempt_counter:
        wrong_move_counter:
        timeout_counter:
        max_attempt_counter:
        max_attempt_per_object:
    Return:
        game_state_counter
        attempt_counter
        correct_move_counter
        wrong_move_counter
        timeout_counter
        max_attempt_counter
    '''

    if task_progress_counter >= 0 and task_progress_counter < 2:
        game_state_counter = 0
    elif task_progress_counter >= 2 and task_progress_counter < 4:
        game_state_counter = 1
    elif task_progress_counter >= 4 and task_progress_counter < 5:
        game_state_counter = 2
    else:
        game_state_counter = 3

    # if then else are necessary to classify the task game state into beg, mid, end

    if user_action == 0 and game_state_counter<3:
        attempt_counter = 1
        correct_move_counter += 1
        task_progress_counter += 1
    # if the user made a wrong move and still did not reach the maximum number of attempts
    elif user_action == 1 and attempt_counter < max_attempt_per_object and game_state_counter<3:
        attempt_counter += 1
        wrong_move_counter += 1
    # if the user did not move any token and still did not reach the maximum number of attempts
    elif user_action == 2 and attempt_counter < max_attempt_per_object and game_state_counter<3:
        attempt_counter += 1
        timeout_counter += 1
    # the agent or therapist makes the correct move on the patient's behalf
    elif attempt_counter>=max_attempt_per_object and game_state_counter<3:
        attempt_counter = 1
        max_attempt_counter += 1
        task_progress_counter +=1

    if game_state_counter==3:
        attempt_counter = 1
        task_progress_counter +=1
        print("Reach the end of the episode")


    next_state = (game_state_counter, attempt_counter, user_action)

    return next_state, task_progress_counter, game_state_counter, attempt_counter, correct_move_counter, wrong_move_counter, timeout_counter, max_attempt_counter


def select_agent_action(agent_action, epsilon):
    '''
    Args:
        agent_action: list of possible actions with their probabilities
    Return:
        one of the agent's actions
    '''

    if random.random()>epsilon:
        return np.argmax(agent_action)
    else:
        agent_action_rm_best = agent_action[:]
        agent_action_rm_best[np.argmax(agent_action)] = 0
        return np.argmax(agent_action_rm_best)

def simulation(bn_model_user_action,
               bn_model_agent_behaviour,
               var_user_action_target_action,
               var_agent_behaviour_target_action,
               game_state_bn_name, attempt_bn_name,
               agent_assistance_bn_name,
               agent_policy,
               state_space, action_space,
               epoch=50,  run = 50, task_complexity=5, max_attempt_per_object=4, alpha_learning=0):
    '''
    Args:

    Return:
        n_correct_per_episode:
        n_wrong_per_episode:
        n_timeout_per_episode:

    '''

    user_action_per_agent_assistance = [[0 for i in range(User_Action.counter.value)]
                                                           for j in range(Agent_Assistance.counter.value)]
    attempt_counter_per_user_action = [[0 for i in range(Attempt.counter.value)] for j in
                                       range(User_Action.counter.value)]
    game_state_counter_per_user_action = [[0 for i in range(Game_State.counter.value)] for j in
                                          range(User_Action.counter.value)]

    #output variables:
    n_correct_per_episode_epoch = [0]*epoch
    n_wrong_per_episode_epoch = [0]*epoch
    n_timeout_per_episode_epoch = [0]*epoch
    n_max_attempt_per_episode_epoch = [0]*epoch
    game_performance_episode_epoch = [0]*epoch
    n_assistance_lev_per_episode_epoch = [[0 for i in range(Agent_Assistance.counter.value)] for j in range(epoch)]


    #data structure to memorise a sequence of episode
    episodes = []
    ep = Episode()
    bn_model_user_action_ref = copy.deepcopy(bn_model_user_action)

    for e in range(epoch):
        print("##########################################################")
        print("EPISODE ", e)
        print("##########################################################")
        bn_model_user_action =  copy.deepcopy(bn_model_user_action_ref)

        n_correct_per_episode_run = [0] * run
        n_wrong_per_episode_run = [0] * run
        n_timeout_per_episode_run = [0] * run
        n_max_attempt_per_episode_run = [0] * run
        game_performance_episode_run = [0] * run
        n_assistance_lev_per_episode_run = [[0 for i in range(Agent_Assistance.counter.value)] for j in range(run)]

        for r in range(run):

            '''Simulation framework'''
            #counters
            game_state_counter = 0
            attempt_counter = 1
            iter_counter = 0
            correct_move_counter = 0
            wrong_move_counter = 0
            timeout_counter = 0
            max_attempt_counter = 0

            #The following variables are used to update the BN at the end of the episode
            user_action_dynamic_variables = {
                                            'attempt': attempt_counter_per_user_action,
                                            'game_state': game_state_counter_per_user_action,
                                            'user_action': user_action_per_agent_assistance
                                            }



            #data structure to memorise the sequence of states  (state, action, next_state)
            episode = []
            selected_user_action = 0
            task_progress_counter = 0
            #####################SIMULATE ONE EPISODE#########################################
            while(task_progress_counter<=task_complexity):

                current_state = (game_state_counter, attempt_counter, selected_user_action)
                current_state_index = ep.state_from_point_to_index(state_space, current_state)
                if agent_policy==[]:
                    vars_agent_evidence = {game_state_bn_name: game_state_counter,
                                          attempt_bn_name: attempt_counter - 1,
                                          }

                    query_agent_behaviour_prob = bn_functions.infer_prob_from_state(user_bn_model=bn_model_agent_behaviour,
                                                                                infer_variable=var_agent_behaviour_target_action,
                                                                                evidence_variables=vars_agent_evidence)

                    #selected_agent_behaviour_action = bn_functions.get_stochastic_action(query_agent_behaviour_prob.values)
                    selected_agent_behaviour_action = select_agent_action(query_agent_behaviour_prob.values, epsilon=0.2)
                else:
                    selected_agent_behaviour_action = select_agent_action(agent_policy[current_state_index], epsilon=0.2)
                    #selected_agent_behaviour_action = bn_functions.get_stochastic_action(agent_policy[current_state_index])
                    #selected_agent_behaviour_action =np.argmax(agent_policy[current_state_index])

                #counters for plots
                n_assistance_lev_per_episode_run[r][selected_agent_behaviour_action] += 1
                print("agent_assistance {},  attempt {}, game {}".format(selected_agent_behaviour_action, attempt_counter, game_state_counter))

                ##########################QUERY FOR THE USER ACTION AND REACT TIME#####################################
                #return the user action in this state based on the Persona profile
                vars_user_evidence = {    game_state_bn_name: game_state_counter,
                                          attempt_bn_name: attempt_counter - 1,
                                          agent_assistance_bn_name: selected_agent_behaviour_action,
                                          }

                query_user_action_prob = bn_functions.infer_prob_from_state(user_bn_model=bn_model_user_action,
                                                                            infer_variable=var_user_action_target_action,
                                                                            evidence_variables=vars_user_evidence)

                selected_user_action = bn_functions.get_stochastic_action(query_user_action_prob.values)
                #selected_user_action = np.argmax(query_user_action_prob.values)

                #updates counters for simulation
                iter_counter += 1
                next_state, task_progress_counter, game_state_counter, attempt_counter, correct_move_counter, \
                wrong_move_counter, timeout_counter, max_attempt_counter = compute_next_state(selected_user_action,
                                                                            task_progress_counter,
                                                                            attempt_counter,
                                                                            correct_move_counter, wrong_move_counter,
                                                                            timeout_counter, max_attempt_counter,
                                                                            max_attempt_per_object)

                # update counters
                if game_state_counter <= 2:
                    user_action_per_agent_assistance[selected_agent_behaviour_action][selected_user_action] += 1
                    attempt_counter_per_user_action[selected_user_action][attempt_counter - 1] += 1
                    game_state_counter_per_user_action[selected_user_action][game_state_counter] += 1

                # store the (state, action, next_state)
                episode.append((ep.state_from_point_to_index(state_space, current_state),
                                selected_agent_behaviour_action,
                                ep.state_from_point_to_index(state_space, next_state)))

                print("current_state ", current_state, " user_action:", selected_user_action, " next_state ", next_state)
            ####################################END of EPISODE#######################################
            print("game_state_counter {}, iter_counter {}, correct_counter {}, wrong_counter {}, "
                  "timeout_counter {}, max_attempt {}".format(game_state_counter, iter_counter, correct_move_counter,
                                                              wrong_move_counter, timeout_counter, max_attempt_counter))

            #save episode
            episodes.append(Episode(episode))

            #update user models
            # bn_model_user_action = bn_functions.update_cpds_tables(bn_model_user_action, user_action_dynamic_variables, alpha_learning)

            #reset counter
            user_action_per_agent_assistance = [[0 for i in range(User_Action.counter.value)]
                                                for j in range(Agent_Assistance.counter.value)]
            attempt_counter_per_user_action = [[0 for i in range(Attempt.counter.value)] for j in
                                               range(User_Action.counter.value)]
            game_state_counter_per_user_action = [[0 for i in range(Game_State.counter.value)] for j in
                                                  range(User_Action.counter.value)]

            #for plots
            n_correct_per_episode_run[r] = correct_move_counter
            n_wrong_per_episode_run[r] = wrong_move_counter
            n_timeout_per_episode_run[r] = timeout_counter
            n_max_attempt_per_episode_run[r] = max_attempt_counter
            game_performance_episode_run[r] = [n_correct_per_episode_run[r],
                                           n_wrong_per_episode_run[r],
                                           n_timeout_per_episode_run[r],
                                           n_max_attempt_per_episode_run[r]]

        #compute average of the values for one epoch and store it
        n_correct_per_episode_epoch[e] = sum(n_correct_per_episode_run)/run
        n_wrong_per_episode_epoch[e] = sum(n_wrong_per_episode_run)/run
        n_timeout_per_episode_epoch[e] = sum(n_timeout_per_episode_run)/run
        n_max_attempt_per_episode_epoch[e] = sum(n_max_attempt_per_episode_run)/run
        game_performance_episode_epoch[e] = list(map(lambda x: sum(x)/run, zip(*game_performance_episode_run)))
        n_assistance_lev_per_episode_epoch[e] = list(map(lambda x: sum(x)/run, zip(*n_assistance_lev_per_episode_run)))

        #reset variables
        n_correct_per_episode_run = [0] * run
        n_wrong_per_episode_run = [0] * run
        n_timeout_per_episode_run = [0] * run
        n_max_attempt_per_episode_run = [0] * run
        game_performance_episode_run = [0] * run
        n_assistance_lev_per_episode_run = [[0 for i in range(Agent_Assistance.counter.value)] for j in range(run)]



    return game_performance_episode_epoch, n_assistance_lev_per_episode_epoch, episodes



#############################################################################
#############################################################################
####################### RUN THE SIMULATION ##################################
#############################################################################
#############################################################################



# # SIMULATION PARAMS
# epochs = 20
# scaling_factor = 1
# # initialise the agent
# bn_model_user_action = bnlearn.import_DAG('/home/pal/Documents/Framework/bn_generative_model/bn_persona_model/persona_model_template.bif')
#
# # initialise memory, attention and reactivity variables
# persona_memory = 0;
# persona_attention = 0;
# persona_reactivity = 1;
#
# # define state space struct for the irl algorithm
# episode_instance = Episode()
# # DEFINITION OF THE MDP
# # define state space struct for the irl algorithm
# attempt = [i for i in range(1, Attempt.counter.value + 1)]
# # +1 (3,_,_) absorbing state
# game_state = [i for i in range(0, Game_State.counter.value + 1)]
# user_action = [i for i in range(-1, User_Action.counter.value - 1)]
# state_space = (game_state, attempt, user_action)
# states_space_list = list(itertools.product(*state_space))
# state_space_index = [episode_instance.state_from_point_to_index(states_space_list, s) for s in states_space_list]
# agent_assistance_action = [i for i in range(Agent_Assistance.counter.value)]
# agent_feedback_action = [i for i in range(Agent_Feedback.counter.value)]
# action_space = (agent_feedback_action, agent_assistance_action)
# action_space_list = list(itertools.product(*action_space))
# action_space_index = [episode_instance.state_from_point_to_index(action_space_list, a) for a in action_space_list]
# terminal_state = [(Game_State.counter.value, i, user_action[j]) for i in range(1, Attempt.counter.value + 1) for j in
#                   range(len(user_action))]
# initial_state = (1, 1, 0)
#
# #1. RUN THE SIMULATION WITH THE PARAMS SET BY THE CAREGIVER
# agent_policy = generate_agent_assistance(preferred_assistance=2, agent_behaviour="help", current_state=5, state_space=states_space_list, action_space=action_space_list)
# print(agent_policy)
#
#
#
# game_performance_per_episode, react_time_per_episode, agent_assistance_per_episode, agent_feedback_per_episode, episodes_list = \
#     simulation(bn_model_user_action=bn_model_user_action, var_user_action_target_action=['user_action'],
#                    game_state_bn_name="game_state",
#                    attempt_bn_name="attempt",
#                    agent_assistance_bn_name="agent_assistance",
#                    agent_feedback_bn_name="agent_feedback",
#                    agent_policy=agent_policy,
#                    state_space=states_space_list, action_space=action_space_list,
#                    epochs=epochs, task_complexity=5, max_attempt_per_object=4, alpha_learning=0.1)
#
# utils.plot2D_game_performance("results/user_performance.png", epochs, scaling_factor, game_performance_per_episode)
# utils.plot2D_assistance("results/agent_assistance.png", epochs, scaling_factor, agent_assistance_per_episode)

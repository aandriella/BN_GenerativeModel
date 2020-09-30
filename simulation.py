import itertools
import os
import bnlearn
import numpy as np
import random
#import classes and modules
from bn_variables import Agent_Assistance, Agent_Feedback, User_Action, User_React_time, Game_State, Attempt
import bn_functions
import utils
from episode import Episode

#
# def choose_next_states(task_progress, game_state_t0, n_attempt_t0, max_attempt_per_object,
#                        selected_agent_assistance_action,
#                        bn_model_user_action, var_user_action_target_action):
#
#     def get_next_state(task_progress, game_state_t0, n_attempt_t0, max_attempt_per_object):
#
#         next_state = []
#
#         #correct move on the last state of the bin
#         if (task_progress == 1 or task_progress == 3 or task_progress == 4) and n_attempt_t0<max_attempt_per_object:
#             next_state.append((game_state_t0+1, n_attempt_t0+1))
#         #correct state bu still in the bin
#         elif task_progress == 0 or task_progress == 2 and n_attempt_t0<max_attempt_per_object:
#             next_state.append((game_state_t0, n_attempt_t0+1))
#         elif (task_progress == 1 or task_progress == 3 or task_progress == 4) and n_attempt_t0>=max_attempt_per_object:
#            assert "you reach the maximum number of attempt the agent will move it for you"
#         elif task_progress == 0 or task_progress == 2 and n_attempt_t0>=max_attempt_per_object:
#             assert "you reach the maximum number of attempt the agent will move it for you"
#
#         return next_state
#
#     next_state = get_next_state(task_progress, game_state_t0, n_attempt_t0, max_attempt_per_object)
#     query_answer_probs = []
#     for t in next_state:
#         vars_user_evidence = {"game_state_t0": game_state_t0,
#                           "attempt_t0": n_attempt_t0 - 1,
#                           "robot_assistance": selected_agent_assistance_action,
#                           "game_state_t1": t[0],
#                           "attempt_t1": t[1],
#                           }
#
#         query_user_action_prob = bn_functions.infer_prob_from_state(bn_model_user_action,
#                                                                 infer_variable=var_user_action_target_action,
#                                                                 evidence_variables=vars_user_evidence)
#         query_answer_probs.append(query_user_action_prob)
#
#
#     #do the inference here
#     #1. check given the current_state which are the possible states
#     #2. for each of the possible states get the probability of user_action
#     #3. select the state with the most higher action and execute it
#     #4. return user_action
#

def generate_agent_assistance(preferred_assistance, agent_behaviour, n_game_state, n_attempt, alpha_action=0.1):
    agent_policy = [[0 for j in range(n_attempt)] for i in range(n_game_state)]
    previous_assistance = -1
    def get_alternative_action(agent_assistance, previous_assistance, agent_behaviour, alpha_action):
        agent_assistance_res = agent_assistance
        if previous_assistance == agent_assistance:
            if agent_behaviour == "challenge":
                if random.random() > alpha_action:
                    agent_assistance_res = min(max(0, agent_assistance-1), 5)
                else:
                    agent_assistance_res = min(max(0, agent_assistance), 5)
            else:
                if random.random() > alpha_action:
                    agent_assistance_res = min(max(0, agent_assistance + 1), 5)
                else:
                    agent_assistance_res = min(max(0, agent_assistance), 5)
        return agent_assistance_res


    for gs in range(n_game_state):
        for att in range(n_attempt):
            if att == 0:
                if random.random()>alpha_action:
                    agent_policy[gs][att] = preferred_assistance
                    previous_assistance = agent_policy[gs][att]
                else:
                    if random.random()>0.5:
                        agent_policy[gs][att] = min(max(0, preferred_assistance-1),5)
                        previous_assistance = agent_policy[gs][att]
                    else:
                        agent_policy[gs][att] = min(max(0, preferred_assistance+1), 5)
                        previous_assistance = agent_policy[gs][att]
            else:
                if agent_behaviour == "challenge":
                    agent_policy[gs][att] = min(max(0, preferred_assistance-1), 5)
                    agent_policy[gs][att] = get_alternative_action(agent_policy[gs][att], previous_assistance, agent_behaviour, alpha_action)
                    previous_assistance = agent_policy[gs][att]
                else:
                    agent_policy[gs][att] = min(max(0, preferred_assistance+1), 5)
                    agent_policy[gs][att] = get_alternative_action(agent_policy[gs][att], previous_assistance, agent_behaviour, alpha_action)
                    previous_assistance = agent_policy[gs][att]

    return agent_policy


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

    if user_action == 1 and game_state_counter<3:
        attempt_counter = 1
        correct_move_counter += 1
        task_progress_counter += 1
    # if the user made a wrong move and still did not reach the maximum number of attempts
    elif user_action == -1 and attempt_counter < max_attempt_per_object and game_state_counter<3:
        attempt_counter += 1
        wrong_move_counter += 1
    # if the user did not move any token and still did not reach the maximum number of attempts
    elif user_action == 0 and attempt_counter < max_attempt_per_object and game_state_counter<3:
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

    # TODO call the function to compute the state of the game (beg, mid, end)




    next_state = (game_state_counter, attempt_counter, user_action)

    return next_state, task_progress_counter, game_state_counter, attempt_counter, correct_move_counter, wrong_move_counter, timeout_counter, max_attempt_counter



def simulation(bn_model_user_action, var_user_action_target_action, bn_model_user_react_time, var_user_react_time_target_action,
               user_memory_name, user_memory_value, user_attention_name, user_attention_value,
               user_reactivity_name, user_reactivity_value,
               task_progress_t0_name, task_progress_t1_name, game_attempt_t0_name, game_attempt_t1_name,
               agent_assistance_name, agent_policy,
               state_space, action_space,
               epochs=50, task_complexity=5, max_attempt_per_object=4, alpha_learning=0):
    '''
    Args:

    Return:
        n_correct_per_episode:
        n_wrong_per_episode:
        n_timeout_per_episode:

    '''
    #TODO: remove agent_assistance_vect and agent_feedback_vect

    #metrics we need, in order to compute afterwords the belief

    agent_feedback_per_action = [[0 for i in range(Agent_Feedback.counter.value)] for j in range(User_Action.counter.value)]
    agent_assistance_per_action = [[0 for i in range(Agent_Assistance.counter.value)] for j in range(User_Action.counter.value)]

    attempt_counter_per_react_time = [[0 for i in range(Attempt.counter.value)] for j in range(User_React_time.counter.value)]
    game_state_counter_per_react_time = [[0 for i in range(Game_State.counter.value)] for j in range(User_React_time.counter.value)]
    agent_feedback_per_react_time = [[0 for i in range(Agent_Feedback.counter.value)] for j in  range(User_React_time.counter.value)]
    agent_assistance_per_react_time = [[0 for i in range(Agent_Assistance.counter.value)] for j in   range(User_React_time.counter.value)]

    game_state_counter_per_agent_feedback = [[0 for i in range(Game_State.counter.value)] for j in   range(Agent_Feedback.counter.value)]
    attempt_counter_per_agent_feedback = [[0 for i in range(Attempt.counter.value)] for j in   range(Agent_Feedback.counter.value)]
    game_state_counter_per_agent_assistance = [[0 for i in range(Game_State.counter.value)] for j in
                                             range(Agent_Assistance.counter.value)]
    attempt_counter_per_agent_assistance = [[0 for i in range(Attempt.counter.value)] for j in
                                          range(Agent_Assistance.counter.value)]


    user_action_per_game_state_attempt_counter_agent_assistance = [[[[0 for i in range(User_Action.counter.value)] for l in range(Game_State.counter.value)] for j in
                                               range(Attempt.counter.value)] for k in range(Agent_Assistance.counter.value)]
    user_action_per_agent_assistance = [[0 for i in range(User_Action.counter.value)] for j in
                                            range(Agent_Assistance.counter.value)]
    attempt_counter_per_user_action = [[0 for i in range(Attempt.counter.value)] for j in range(User_Action.counter.value)]
    game_state_counter_per_user_action = [[0 for i in range(Game_State.counter.value)] for j in
                                     range(User_Action.counter.value)]

    #output variables:
    n_correct_per_episode = [0]*epochs
    n_wrong_per_episode = [0]*epochs
    n_timeout_per_episode = [0]*epochs
    n_max_attempt_per_episode = [0]*epochs
    game_performance_episode = [0]*epochs
    n_assistance_lev_per_episode = [[0 for i in range(Agent_Assistance.counter.value)] for j in range(epochs)]
    n_feedback_per_episode = [[0 for i in range(Agent_Feedback.counter.value)] for j in range(epochs)]
    n_react_time_per_episode = [[0 for i in range(User_React_time.counter.value)] for j in range(epochs)]


    #data structure to memorise a sequence of episode
    episodes = []
    ep = Episode()

    for e in range(epochs):
        print("##########################################################")
        print("EPISODE ",e)
        print("##########################################################")

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
                                        'attempt_t1': attempt_counter_per_user_action,
                                        'game_state_t1': game_state_counter_per_user_action,
                                        'user_action': user_action_per_game_state_attempt_counter_agent_assistance
                                        }

        user_react_time_dynamic_variables = {'attempt': attempt_counter_per_react_time,
                             'game_state': game_state_counter_per_react_time,
                             'agent_assistance': agent_assistance_per_react_time,
                             'agent_feedback': agent_feedback_per_react_time}

        agent_assistance_dynamic_variables = {'attempt': attempt_counter_per_agent_assistance,
                                  'game_state': game_state_counter_per_agent_assistance}

        agent_feedback_dynamic_variables = {'attempt': attempt_counter_per_agent_feedback,
                                  'game_state': game_state_counter_per_agent_feedback}

        #data structure to memorise the sequence of states  (state, action, next_state)
        episode = []
        selected_user_action = 0
        task_progress_counter = 0
        #####################SIMULATE ONE EPISODE#########################################
        while(task_progress_counter<=task_complexity):

            current_state = (game_state_counter, attempt_counter, selected_user_action)

            selected_agent_assistance_action = agent_policy[game_state_counter][attempt_counter-1]#random.randint(0,5)
            selected_agent_feedback_action = 0#random.randint(0,1)

            #counters for plots
            n_assistance_lev_per_episode[e][selected_agent_assistance_action] += 1
            current_agent_action = (selected_agent_feedback_action, selected_agent_assistance_action)

            print("agent_assistance {}, attempt {}, game {}, agent_feedback {}".format(selected_agent_assistance_action, attempt_counter, game_state_counter, selected_agent_feedback_action))


            ##########################QUERY FOR THE USER ACTION AND REACT TIME#####################################
            #compare the real user with the estimated Persona and returns a user action (0, 1a, 2)

            #return the user action in this state based on the Persona profile
            vars_user_evidence = {    task_progress_t0_name: game_state_counter,
                                      game_attempt_t0_name: attempt_counter - 1,
                                      task_progress_t1_name: game_state_counter,
                                      game_attempt_t1_name: attempt_counter - 1,
                                      agent_assistance_name: selected_agent_assistance_action,
                                      }

            query_user_action_prob = bn_functions.infer_prob_from_state(bn_model_user_action,
                                                                        infer_variable=var_user_action_target_action,
                                                                        evidence_variables=vars_user_evidence)
            # query_user_react_time_prob = bn_functions.infer_prob_from_state(bn_model_user_react_time,
            #                                                                 infer_variable=var_user_react_time_target_action,
            #                                                                 evidence_variables=vars_user_evidence)
            #
            #

            selected_user_action = bn_functions.get_stochastic_action(query_user_action_prob.values)
            # selected_user_react_time = bn_functions.get_stochastic_action(query_user_react_time_prob.values)
            # counters for plots
            # n_react_time_per_episode[e][selected_user_react_time] += 1

            #updates counters for user action

            user_action_per_game_state_attempt_counter_agent_assistance[selected_agent_assistance_action][attempt_counter-1][game_state_counter][selected_user_action] += 1
            attempt_counter_per_user_action[selected_user_action][attempt_counter-1] += 1
            game_state_counter_per_user_action[selected_user_action][game_state_counter] += 1
            user_action_per_agent_assistance[selected_agent_assistance_action][selected_user_action] += 1

            #update counter for user react time
            # agent_assistance_per_react_time[selected_user_react_time][selected_agent_assistance_action] += 1
            # attempt_counter_per_react_time[selected_user_react_time][attempt_counter-1] += 1
            # game_state_counter_per_react_time[selected_user_react_time][game_state_counter] += 1
            # agent_feedback_per_react_time[selected_user_react_time][selected_agent_feedback_action] += 1
            #update counter for agent feedback
            game_state_counter_per_agent_feedback[selected_agent_feedback_action][game_state_counter] += 1
            attempt_counter_per_agent_feedback[selected_agent_feedback_action][attempt_counter-1] += 1
            #update counter for agent assistance
            game_state_counter_per_agent_assistance[selected_agent_assistance_action][game_state_counter] += 1
            attempt_counter_per_agent_assistance[selected_agent_assistance_action][attempt_counter-1] += 1

            # updates counters for simulation
            # remap user_action index
            if selected_user_action == 0:
              selected_user_action = 1
            elif selected_user_action == 1:
              selected_user_action = -1
            else:
              selected_user_action = 0

            #updates counters for simulation
            iter_counter += 1
            next_state, task_progress_counter, game_state_counter, attempt_counter, correct_move_counter, \
            wrong_move_counter, timeout_counter, max_attempt_counter = compute_next_state(selected_user_action,
                                                                        task_progress_counter,
                                                                        attempt_counter,
                                                                        correct_move_counter, wrong_move_counter,
                                                                        timeout_counter, max_attempt_counter,
                                                                        max_attempt_per_object)

            # store the (state, action, next_state)
            episode.append((ep.state_from_point_to_index(state_space, current_state),
                            ep.state_from_point_to_index(action_space, current_agent_action),
                            ep.state_from_point_to_index(state_space, next_state)))

            print("current_state ", current_state, " next_state ", next_state)
        ####################################END of EPISODE#######################################
        print("game_state_counter {}, iter_counter {}, correct_counter {}, wrong_counter {}, "
              "timeout_counter {}, max_attempt {}".format(game_state_counter, iter_counter, correct_move_counter,
                                                          wrong_move_counter, timeout_counter, max_attempt_counter))

        #save episode
        episodes.append(Episode(episode))

        #update user models
        bn_model_user_action = bn_functions.update_cpds_tables(bn_model_user_action, user_action_dynamic_variables, alpha_learning)
        bn_model_user_react_time = bn_functions.update_cpds_tables(bn_model_user_react_time, user_react_time_dynamic_variables)
        #update agent models

        print("user_given_game_attempt:", bn_model_user_action['model'].cpds[0].values)
        print("user_given_robot:", bn_model_user_action['model'].cpds[5].values)
        print("game_user:", bn_model_user_action['model'].cpds[3].values)
        print("attempt_user:", bn_model_user_action['model'].cpds[2].values)

        #reset counter
        user_action_per_game_state_attempt_counter_agent_assistance = [[[[0 for i in range(User_Action.counter.value)]
                                                                         for l in range(Game_State.counter.value)] for j in
                                                                        range(Attempt.counter.value)] for k in range(Agent_Assistance.counter.value)]
        user_action_per_agent_assistance = [[0 for i in range(User_Action.counter.value)] for j in
                                            range(Agent_Assistance.counter.value)]
        attempt_counter_per_user_action = [[0 for i in range(Attempt.counter.value)] for j in
                                           range(User_Action.counter.value)]
        game_state_counter_per_user_action = [[0 for i in range(Game_State.counter.value)] for j in
                                              range(User_Action.counter.value)]

        attempt_counter_per_react_time = [[0 for i in range(Attempt.counter.value)] for j in
                                          range(User_React_time.counter.value)]
        game_state_counter_per_react_time = [[0 for i in range(Game_State.counter.value)] for j in
                                             range(User_React_time.counter.value)]
        agent_feedback_per_react_time = [[0 for i in range(Agent_Feedback.counter.value)] for j in
                                         range(User_React_time.counter.value)]
        agent_assistance_per_react_time = [[0 for i in range(Agent_Assistance.counter.value)] for j in
                                           range(User_React_time.counter.value)]


        #for plots
        n_correct_per_episode[e] = correct_move_counter
        n_wrong_per_episode[e] = wrong_move_counter
        n_timeout_per_episode[e] = timeout_counter
        n_max_attempt_per_episode[e] = max_attempt_counter
        game_performance_episode[e] = [n_correct_per_episode[e],
                                       n_wrong_per_episode[e],
                                       n_timeout_per_episode[e],
                                       n_max_attempt_per_episode[e]]


    return game_performance_episode, n_react_time_per_episode, n_assistance_lev_per_episode, n_feedback_per_episode, episodes



#############################################################################
#############################################################################
####################### RUN THE SIMULATION ##################################
#############################################################################
#############################################################################


agent_policy = generate_agent_assistance(preferred_assistance=2, agent_behaviour="help", n_game_state=Game_State.counter.value, n_attempt=Attempt.counter.value, alpha_action=0.5)

# SIMULATION PARAMS
epochs = 20
scaling_factor = 1
# initialise the agent
bn_model_caregiver_assistance = bnlearn.import_DAG('/home/pal/Documents/Framework/bn_generative_model/bn_agent_model/agent_assistive_model.bif')
bn_model_caregiver_feedback = None#bnlearn.import_DAG('/home/pal/Documents/Framework/bn_generative_model/bn_agent_model/agent_feedback_model.bif')
bn_model_user_action = bnlearn.import_DAG('/home/pal/Documents/Framework/bn_generative_model/bn_persona_model/persona_model_test.bif')
bn_model_user_react_time = bnlearn.import_DAG('/home/pal/Documents/Framework/bn_generative_model/bn_persona_model/user_react_time_model.bif')

# initialise memory, attention and reactivity variables
persona_memory = 0;
persona_attention = 0;
persona_reactivity = 1;

# define state space struct for the irl algorithm
episode_instance = Episode()
# DEFINITION OF THE MDP
# define state space struct for the irl algorithm
attempt = [i for i in range(1, Attempt.counter.value + 1)]
# +1 (3,_,_) absorbing state
game_state = [i for i in range(0, Game_State.counter.value + 1)]
user_action = [i for i in range(-1, User_Action.counter.value - 1)]
state_space = (game_state, attempt, user_action)
states_space_list = list(itertools.product(*state_space))
state_space_index = [episode_instance.state_from_point_to_index(states_space_list, s) for s in states_space_list]
agent_assistance_action = [i for i in range(Agent_Assistance.counter.value)]
agent_feedback_action = [i for i in range(Agent_Feedback.counter.value)]
action_space = (agent_feedback_action, agent_assistance_action)
action_space_list = list(itertools.product(*action_space))
action_space_index = [episode_instance.state_from_point_to_index(action_space_list, a) for a in action_space_list]
terminal_state = [(Game_State.counter.value, i, user_action[j]) for i in range(1, Attempt.counter.value + 1) for j in
                  range(len(user_action))]
initial_state = (1, 1, 0)

#1. RUN THE SIMULATION WITH THE PARAMS SET BY THE CAREGIVER


game_performance_per_episode, react_time_per_episode, agent_assistance_per_episode, agent_feedback_per_episode, episodes_list = \
    simulation(bn_model_user_action=bn_model_user_action, var_user_action_target_action=['user_action'],
                   bn_model_user_react_time=bn_model_user_react_time,
                   var_user_react_time_target_action=['user_react_time'],
                   user_memory_name="memory", user_memory_value=persona_memory,
                   user_attention_name="attention", user_attention_value=persona_attention,
                   user_reactivity_name="reactivity", user_reactivity_value=persona_reactivity,
                   task_progress_t0_name="game_state_t0", task_progress_t1_name="game_state_t1",
                   game_attempt_t0_name="attempt_t0", game_attempt_t1_name="attempt_t1",
                   agent_assistance_name="agent_assistance", agent_policy=agent_policy,
                   state_space=states_space_list, action_space=action_space_list,
                   epochs=epochs, task_complexity=5, max_attempt_per_object=4, alpha_learning=0.1)

utils.plot2D_game_performance("/home/pal/Documents/Framework/bn_generative_model/results/user_performance.png", epochs, scaling_factor, game_performance_per_episode)
utils.plot2D_assistance("/home/pal/Documents/Framework/bn_generative_model/results/agent_assistance.png", epochs, scaling_factor, agent_assistance_per_episode)

import itertools
import os
import bnlearn
import numpy as np
#import classes and modules
from bn_variables import Robot_Assistance, Robot_Feedback, User_Action, User_React_time, Game_State, Attempt
import bn_functions
import utils
import episode as ep


def compute_next_state(user_action, task_progress_counter, attempt_counter, correct_move_counter,
                       wrong_move_counter, timeout_counter, max_attempt_counter, max_attempt_per_object
                       ):
    '''
    The function computes given the current state and action of the user, the next state
    Args:
        user_action: -1 wrong, 0 timeout, 1 correct
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

    # if then else are necessary to classify the task game state into beg, mid, end

    if user_action == 1:
      attempt_counter = 1
      correct_move_counter += 1
      task_progress_counter += 1
    # if the user made a wrong move and still did not reach the maximum number of attempts
    elif user_action == -1 and attempt_counter < max_attempt_per_object:
      attempt_counter += 1
      wrong_move_counter += 1
    # if the user did not move any token and still did not reach the maximum number of attempts
    elif user_action == 0 and attempt_counter < max_attempt_per_object:
      attempt_counter += 1
      timeout_counter += 1
    # the robot or therapist makes the correct move on the patient's behalf
    else:
      attempt_counter = 1
      max_attempt_counter += 1
      task_progress_counter +=1
      correct_move_counter += 1


    # TODO call the function to compute the state of the game (beg, mid, end)

    if correct_move_counter >= 0 and correct_move_counter <= 2:
      game_state_counter = 0
    elif correct_move_counter > 2 and correct_move_counter <= 4:
      game_state_counter = 1
    elif correct_move_counter>4 and correct_move_counter<=5:
      game_state_counter = 2
    else:
      game_state_counter = 3

    next_state = (game_state_counter, attempt_counter, user_action)

    return next_state, task_progress_counter, game_state_counter, attempt_counter, correct_move_counter, wrong_move_counter, timeout_counter, max_attempt_counter



def simulation(bn_model_user_action, var_user_action_target_action, bn_model_user_react_time, var_user_react_time_target_action,
               user_memory_name, user_memory_value, user_attention_name, user_attention_value,
               user_reactivity_name, user_reactivity_value,
               task_progress_name, game_attempt_name, robot_assistance_name, robot_feedback_name,
               bn_model_robot_assistance, var_robot_assistance_target_action, bn_model_robot_feedback,
               var_robot_feedback_target_action,
               bn_model_other_user_action, var_other_user_action_target_action,
               bn_model_other_user_react_time, var_other_user_target_react_time_action,
               other_user_memory_name, other_user_memory_value,
               other_user_attention_name, other_user_attention_value, other_user_reactivity_name,
               other_user_reactivity_value,
               state_space, action_space,
               epochs=50, task_complexity=5, max_attempt_per_object=4):
    '''
    Args:

    Return:
        n_correct_per_episode:
        n_wrong_per_episode:
        n_timeout_per_episode:

    '''
    #TODO: remove robot_assistance_vect and robot_feedback_vect

    #metrics we need, in order to compute afterwords the belief

    attempt_counter_per_action = [[0 for i in range(Attempt.counter.value)]  for j in range(User_Action.counter.value)]
    game_state_counter_per_action = [[0 for i in range(Game_State.counter.value)]  for j in range(User_Action.counter.value)]
    robot_feedback_per_action = [[0 for i in range(Robot_Feedback.counter.value)] for j in range(User_Action.counter.value)]
    robot_assistance_per_action = [[0 for i in range(Robot_Assistance.counter.value)] for j in range(User_Action.counter.value)]

    attempt_counter_per_react_time = [[0 for i in range(Attempt.counter.value)] for j in range(User_React_time.counter.value)]
    game_state_counter_per_react_time = [[0 for i in range(Game_State.counter.value)] for j in range(User_React_time.counter.value)]
    robot_feedback_per_react_time = [[0 for i in range(Robot_Feedback.counter.value)] for j in  range(User_React_time.counter.value)]
    robot_assistance_per_react_time = [[0 for i in range(Robot_Assistance.counter.value)] for j in   range(User_React_time.counter.value)]

    game_state_counter_per_robot_assistance = [[0 for i in range(Game_State.counter.value)] for j in   range(Robot_Assistance.counter.value)]
    attempt_counter_per_robot_assistance = [[0 for i in range(Attempt.counter.value)] for j in   range(Robot_Assistance.counter.value)]

    game_state_counter_per_robot_feedback = [[0 for i in range(Game_State.counter.value)] for j in   range(Robot_Feedback.counter.value)]
    attempt_counter_per_robot_feedback = [[0 for i in range(Attempt.counter.value)] for j in   range(Robot_Feedback.counter.value)]


    #output variables:
    n_correct_per_episode = [0]*epochs
    n_wrong_per_episode = [0]*epochs
    n_timeout_per_episode = [0]*epochs
    n_max_attempt_per_episode = [0]*epochs
    game_performance_episode = [0]*epochs
    n_assistance_lev_per_episode = [[0 for i in range(Robot_Assistance.counter.value)] for j in range(epochs)]
    n_feedback_per_episode = [[0 for i in range(Robot_Feedback.counter.value)] for j in range(epochs)]
    n_react_time_per_episode = [[0 for i in range(User_React_time.counter.value)] for j in range(epochs)]


    #data structure to memorise a sequence of episode
    episodes = []


    for e in range(epochs):
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
        user_action_dynamic_variables = {'attempt': attempt_counter_per_action,
                             'game_state': game_state_counter_per_action,
                             'robot_assistance': robot_assistance_per_action,
                             'robot_feedback': robot_feedback_per_action}

        user_react_time_dynamic_variables = {'attempt': attempt_counter_per_react_time,
                             'game_state': game_state_counter_per_react_time,
                             'robot_assistance': robot_assistance_per_react_time,
                             'robot_feedback': robot_feedback_per_react_time}

        robot_assistance_dynamic_variables = {'attempt': attempt_counter_per_robot_assistance,
                                  'game_state': game_state_counter_per_robot_assistance}

        robot_feedback_dynamic_variables = {'attempt': attempt_counter_per_robot_feedback,
                                  'game_state': game_state_counter_per_robot_feedback}

        #data structure to memorise the sequence of states  (state, action, next_state)
        episode = []
        selected_user_action = 0
        task_progress_counter = 0
        #####################SIMULATE ONE EPISODE#########################################
        while(task_progress_counter<=task_complexity):

            current_state = (game_state_counter, attempt_counter, selected_user_action)

            ##################QUERY FOR THE ROBOT ASSISTANCE AND FEEDBACK##################
            vars_robot_evidence = {
                                   user_reactivity_name: user_reactivity_value,
                                   user_memory_name: user_memory_value,
                                   task_progress_name: game_state_counter,
                                   game_attempt_name: attempt_counter-1,
                                   }
            query_robot_assistance_prob = bn_functions.infer_prob_from_state(bn_model_robot_assistance,
                                                                   infer_variable=var_robot_assistance_target_action,
                                                                   evidence_variables=vars_robot_evidence)

            query_robot_feedback_prob = bn_functions.infer_prob_from_state(bn_model_robot_feedback,
                                                                      infer_variable=var_robot_feedback_target_action,
                                                                      evidence_variables=vars_robot_evidence)


            selected_robot_assistance_action = bn_functions.get_stochastic_action(query_robot_assistance_prob.values)
            selected_robot_feedback_action = bn_functions.get_stochastic_action(query_robot_feedback_prob.values)

            #counters for plots
            n_assistance_lev_per_episode[e][selected_robot_assistance_action] += 1
            n_feedback_per_episode[e][selected_robot_feedback_action] += 1
            current_robot_action = (selected_robot_assistance_action, selected_robot_feedback_action)

            print("robot_assistance {}, attempt {}, game {}, robot_feedback {}".format(selected_robot_assistance_action, attempt_counter, game_state_counter, selected_robot_feedback_action))


            ##########################QUERY FOR THE USER ACTION AND REACT TIME#####################################
            #compare the real user with the estimated Persona and returns a user action (0, 1, 2)
            if bn_model_other_user_action!=None and bn_model_user_react_time!=None:
                #return the user action in this state based on the user profile
                vars_other_user_evidence = {other_user_attention_name:other_user_attention_value,
                                            other_user_reactivity_name:other_user_reactivity_value,
                                            other_user_memory_name:other_user_memory_value,
                                            task_progress_name:game_state_counter,
                                            game_attempt_name:attempt_counter-1,
                                            robot_assistance_name:selected_robot_assistance_action,
                                            robot_feedback_name:selected_robot_feedback_action
                                            }
                query_user_action_prob = bn_functions.infer_prob_from_state(bn_model_other_user_action,
                                                                            infer_variable=var_other_user_action_target_action,
                                                                            evidence_variables=vars_other_user_evidence)
                query_user_react_time_prob = bn_functions.infer_prob_from_state(bn_model_other_user_react_time,
                                                                                infer_variable=var_other_user_target_react_time_action,
                                                                                evidence_variables=vars_other_user_evidence)


            else:
                #return the user action in this state based on the Persona profile
                vars_user_evidence = {user_attention_name: user_attention_value,
                                      user_reactivity_name: user_reactivity_value,
                                      user_memory_name: user_memory_value,
                                      task_progress_name: game_state_counter,
                                      game_attempt_name: attempt_counter-1,
                                      robot_assistance_name: selected_robot_assistance_action,
                                      robot_feedback_name: selected_robot_feedback_action
                                      }
                query_user_action_prob = bn_functions.infer_prob_from_state(bn_model_user_action,
                                                                       infer_variable=var_user_action_target_action,
                                                                       evidence_variables=vars_user_evidence)
                query_user_react_time_prob = bn_functions.infer_prob_from_state(bn_model_user_react_time,
                                                                       infer_variable=var_user_react_time_target_action,
                                                                       evidence_variables=vars_user_evidence)



            selected_user_action = bn_functions.get_stochastic_action(query_user_action_prob.values)
            selected_user_react_time = bn_functions.get_stochastic_action(query_user_react_time_prob.values)
            # counters for plots
            n_react_time_per_episode[e][selected_user_react_time] += 1

            #updates counters for user action
            robot_assistance_per_action[selected_user_action][selected_robot_assistance_action] += 1
            attempt_counter_per_action[selected_user_action][attempt_counter-1] += 1
            game_state_counter_per_action[selected_user_action][game_state_counter] += 1
            robot_feedback_per_action[selected_user_action][selected_robot_feedback_action] += 1
            #update counter for user react time
            robot_assistance_per_react_time[selected_user_react_time][selected_robot_assistance_action] += 1
            attempt_counter_per_react_time[selected_user_react_time][attempt_counter-1] += 1
            game_state_counter_per_react_time[selected_user_react_time][game_state_counter] += 1
            robot_feedback_per_react_time[selected_user_react_time][selected_robot_feedback_action] += 1
            #update counter for robot feedback
            game_state_counter_per_robot_feedback[selected_robot_feedback_action][game_state_counter] += 1
            attempt_counter_per_robot_feedback[selected_robot_feedback_action][attempt_counter-1] += 1
            #update counter for robot assistance
            game_state_counter_per_robot_assistance[selected_robot_assistance_action][game_state_counter] += 1
            attempt_counter_per_robot_assistance[selected_robot_assistance_action][attempt_counter-1] += 1

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
            episode.append((ep.point_to_index(current_state, state_space),
                            ep.point_to_index(current_robot_action, action_space),
                            ep.point_to_index(next_state, state_space)))

            print("current_state ", current_state, " next_state ", next_state)
        ####################################END of EPISODE#######################################
        print("task_evolution {}, attempt_counter {}, correct_counter {}, wrong_counter {}, timeout_counter {}".format(game_state_counter, iter_counter, correct_move_counter, wrong_move_counter, timeout_counter))
        print("robot_assistance_per_action {}".format(robot_assistance_per_action))
        print("attempt_counter_per_action {}".format(attempt_counter_per_action))
        print("game_state_counter_per_action {}".format(game_state_counter_per_action))
        print("robot_feedback_per_action {}".format(robot_feedback_per_action))
        print("iter {}, correct {}, wrong {}, timeout {}".format(iter_counter, correct_move_counter, wrong_move_counter, timeout_counter))

        #save episode
        episodes.append(episode)

        #update user models
        bn_model_user_action = bn_functions.update_cpds_tables(bn_model_user_action, user_action_dynamic_variables)
        bn_model_user_react_time = bn_functions.update_cpds_tables(bn_model_user_react_time, user_react_time_dynamic_variables)
        #update robot models
        bn_model_robot_assistance = bn_functions.update_cpds_tables(bn_model_robot_assistance, robot_assistance_dynamic_variables)
        bn_model_robot_feedback = bn_functions.update_cpds_tables(bn_model_robot_feedback, robot_feedback_dynamic_variables)

        #reset counter
        robot_assistance_per_action = [[0 for i in range(Robot_Assistance.counter.value)] for j in
                                         range(User_Action.counter.value)]
        robot_feedback_per_action = [[0 for i in range(Robot_Feedback.counter.value)] for j in
                                     range(User_Action.counter.value)]
        game_state_counter_per_action = [[0 for i in range(Game_State.counter.value)] for j in
                                         range(User_Action.counter.value)]
        attempt_counter_per_action = [[0 for i in range(Attempt.counter.value)] for j in
                                      range(User_Action.counter.value)]

        attempt_counter_per_react_time = [[0 for i in range(Attempt.counter.value)] for j in
                                          range(User_React_time.counter.value)]
        game_state_counter_per_react_time = [[0 for i in range(Game_State.counter.value)] for j in
                                             range(User_React_time.counter.value)]
        robot_feedback_per_react_time = [[0 for i in range(Robot_Feedback.counter.value)] for j in
                                         range(User_React_time.counter.value)]
        robot_assistance_per_react_time = [[0 for i in range(Robot_Assistance.counter.value)] for j in
                                           range(User_React_time.counter.value)]

        game_state_counter_per_robot_assistance = [[0 for i in range(Game_State.counter.value)] for j in
                                                   range(Robot_Assistance.counter.value)]
        attempt_counter_per_robot_assistance = [[0 for i in range(Attempt.counter.value)] for j in
                                                range(Robot_Assistance.counter.value)]

        game_state_counter_per_robot_feedback = [[0 for i in range(Game_State.counter.value)] for j in
                                                 range(Robot_Feedback.counter.value)]
        attempt_counter_per_robot_feedback = [[0 for i in range(Attempt.counter.value)] for j in
                                              range(Robot_Feedback.counter.value)]

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

#SIMULATION PARAMS

epochs = 100

#initialise the robot
bn_model_robot_assistance = bnlearn.import_DAG('bn_robot_model/robot_assistive_model.bif')
bn_model_robot_feedback = bnlearn.import_DAG('bn_robot_model/robot_feedback_model.bif')
bn_model_user_action = bnlearn.import_DAG('bn_persona_model/user_action_model.bif')
bn_model_user_react_time = bnlearn.import_DAG('bn_persona_model/user_react_time_model.bif')
bn_model_other_user_action = None#bnlearn.import_DAG('bn_persona_model/other_user_action_model.bif')
bn_model_other_user_react_time = None#bnlearn.import_DAG('bn_persona_model/other_user_react_time_model.bif')

#initialise memory, attention and reactivity varibles
persona_memory = 0; persona_attention = 0; persona_reactivity = 1;
#initialise memory, attention and reactivity varibles
other_user_memory = 2; other_user_attention = 2; other_user_reactivity = 2;

#define state space struct for the irl algorithm
attempt = [i for i in range(1, Attempt.counter.value+1)]
#+1 (3,_,_) absorbing state
game_state = [i for i in range(0, Game_State.counter.value+1)]
user_action = [i for i in range(-1, User_Action.counter.value-1)]
state_space = (game_state, attempt, user_action)
states_space_list = list(itertools.product(*state_space))
robot_assistance_action = [i for i in range(Robot_Assistance.counter.value)]
robot_feedback_action = [i for i in range(Robot_Feedback.counter.value)]
action_space = (robot_assistance_action, robot_feedback_action)
action_space_list = list(itertools.product(*action_space))

game_performance_per_episode, react_time_per_episode, robot_assistance_per_episode, robot_feedback_per_episode, generated_episodes = \
simulation(bn_model_user_action=bn_model_user_action, var_user_action_target_action=['user_action'],
           bn_model_user_react_time=bn_model_user_react_time, var_user_react_time_target_action=['user_react_time'],
           user_memory_name="memory", user_memory_value=persona_memory,
           user_attention_name="attention", user_attention_value=persona_attention,
           user_reactivity_name="reactivity", user_reactivity_value=persona_reactivity,
           task_progress_name="game_state", game_attempt_name="attempt",
           robot_assistance_name="robot_assistance", robot_feedback_name="robot_feedback",
           bn_model_robot_assistance=bn_model_robot_assistance, var_robot_assistance_target_action=["robot_assistance"],
           bn_model_robot_feedback=bn_model_robot_feedback, var_robot_feedback_target_action=["robot_feedback"],
           bn_model_other_user_action=bn_model_other_user_action, var_other_user_action_target_action=['user_action'],
           bn_model_other_user_react_time=bn_model_other_user_react_time,
            var_other_user_target_react_time_action=["user_react_time"], other_user_memory_name="memory",
           other_user_memory_value=other_user_memory, other_user_attention_name="attention",
           other_user_attention_value=other_user_attention, other_user_reactivity_name="reactivity",
           other_user_reactivity_value=other_user_reactivity,
           state_space=states_space_list, action_space=action_space_list,
           epochs=epochs, task_complexity=5, max_attempt_per_object=4)

plot_game_performance_path = ""
plot_robot_assistance_path = ""
episodes_path = "episodes.npy"

if bn_model_other_user_action != None:
    plot_game_performance_path = "game_performance_"+"_epoch_"+str(epochs)+"_real_user_memory_"+str(real_user_memory)+"_real_user_attention_"+str(real_user_attention)+"_real_user_reactivity_"+str(real_user_reactivity)+".jpg"
    plot_robot_assistance_path = "robot_assistance_"+"epoch_"+str(epochs)+"_real_user_memory_"+str(real_user_memory)+"_real_user_attention_"+str(real_user_attention)+"_real_user_reactivity_"+str(real_user_reactivity)+".jpg"
    plot_robot_feedback_path = "robot_feedback_"+"epoch_"+str(epochs)+"_real_user_memory_"+str(real_user_memory)+"_real_user_attention_"+str(real_user_attention)+"_real_user_reactivity_"+str(real_user_reactivity)+".jpg"

else:
    plot_game_performance_path = "game_performance_"+"epoch_" + str(epochs) + "_persona_memory_" + str(persona_memory) + "_persona_attention_" + str(persona_attention) + "_persona_reactivity_" + str(persona_reactivity) + ".jpg"
    plot_robot_assistance_path = "robot_assistance_"+"epoch_"+str(epochs)+"_persona_memory_"+str(persona_memory)+"_persona_attention_"+str(persona_attention)+"_persona_reactivity_"+str(persona_reactivity)+".jpg"
    plot_robot_feedback_path = "robot_feedback_"+"epoch_"+str(epochs)+"_persona_memory_"+str(persona_memory)+"_persona_attention_"+str(persona_attention)+"_persona_reactivity_"+str(persona_reactivity)+".jpg"

dir_name = input("Please insert the name of the directory:")
full_path = os.getcwd()+"/results/"+dir_name+"/"
if not os.path.exists(full_path):
  os.mkdir(full_path)
  print("Directory ", full_path, " created.")
else:
  dir_name = input("The directory already exist please insert a new name:")
  print("Directory ", full_path, " created.")
  if os.path.exists(full_path):
    assert("Directory already exists ... start again")
    exit(0)

with open(full_path+episodes_path, "ab") as f:
  np.save(full_path+episodes_path, generated_episodes)
  f.close()


utils.plot2D_game_performance(full_path+plot_game_performance_path, epochs, game_performance_per_episode)
utils.plot2D_assistance(full_path+plot_robot_assistance_path, epochs, robot_assistance_per_episode)
utils.plot2D_feedback(full_path+plot_robot_feedback_path, epochs, robot_feedback_per_episode)



'''
With the current simulator we can generate a list of episodes
the episodes will be used to generate the trans probabilities and as input to the IRL algo 
'''
#TODO
# - include reaction time as output
# - average mistakes, average timeout, average assistance, average_react_time
# - include real time episodes into the simulation:
#   - counters for robot_assistance, robot_feedback, attempt, game_state, attention and reactivity
#   - using the function update probability to generate the new user model and use it as input to the simulator




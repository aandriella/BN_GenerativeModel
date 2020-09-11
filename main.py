import bnlearn
import os
#import classes and modules
from bn_variables import Memory, Attention, Reactivity, Robot_Assistance, Robot_Feedback, Robot_Assistance_Feedback, User_Action, User_Capability, Game_State, Attempt
import bn_functions
import utils

def compute_next_state(user_action, task_evolution, attempt_counter, correct_move_counter,
                       wrong_move_counter, timeout_counter, max_attept_counter
                       ):
    '''
    The function computes given the current state and action of the user, the next state
    Args:
        user_action: 0,1,2
        task_evolution: beg, mid, end
        correct_move_counter:
        attempt_counter:
        wrong_move_counter:
        timeout_counter:
        max_attempt_counter:
    Return:
        task_evolution
        attempt_counter
        correct_move_counter
        wrong_move_counter
        timeout_counter
        max_attempt_counter
    '''
    if user_action == 0:
        attempt_counter = 0
        task_evolution += 1
        correct_move_counter += 1
    # if the user made a wrong move and still did not reach the maximum number of attempts
    elif user_action == 1 and attempt_counter < 3:
        attempt_counter += 1
        wrong_move_counter += 1
    # if the user did not move any token and still did not reach the maximum number of attempts
    elif user_action == 2 and attempt_counter < 3:
        attempt_counter += 1
        timeout_counter += 1
    # the robot or therapist makes the correct move on the patient's behalf
    else:
        attempt_counter = 0
        task_evolution += 1
        #correct_move_counter += 1
        max_attept_counter += 1

    return task_evolution, attempt_counter, correct_move_counter, wrong_move_counter, timeout_counter, max_attept_counter


def interpret_user_output(action_id):
    user_action = 0
    user_react_time = 0

    if action_id == 0:
        user_action = 0;
        user_react_time = 0
    elif action_id == 1:
        user_action = 1;
        user_react_time = 0
    elif action_id == 2:
        user_action = 2;
        user_react_time = 0
    elif action_id == 3:
        user_action = 0;
        user_react_time = 1
    elif action_id == 4:
        user_action = 1;
        user_react_time = 1
    elif action_id == 5:
        user_action = 2;
        user_react_time = 1
    elif action_id == 6:
        user_action = 0;
        user_react_time = 2
    elif action_id == 7:
        user_action = 1;
        user_react_time = 2
    elif action_id == 8:
        user_action = 2;
        user_react_time = 2

    return user_action, user_react_time

def simulation(user_bn_model, user_var_target, user_memory_name, user_memory_value, user_attention_name, user_attention_value,
               user_reactivity_name, user_reactivity_value,
               task_progress_name, game_attempt_name, robot_assistance_name, robot_feedback_name,
               robot_bn_model, robot_var_target,
               other_user_bn_model, other_user_var_target, other_user_memory_name, other_user_memory_value,
               other_user_attention_name, other_user_attention_value,
               other_user_reactivity_name, other_user_reactivity_value,
               epochs=50, task_complexity=5):
    '''
    Args:

    Return:
        n_correct_per_episode:
        n_wrong_per_episode:
        n_timeout_per_episode:

    '''
    #TODO: remove robot_assistance_vect and robot_feedback_vect

    #metrics we need, in order to compute afterwords the belief
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
    robot_assistance_per_action = [[0 for i in range(Robot_Assistance.counter.value)] for j in range(User_Action.counter.value)]


    #these are the variables of the persona bn that are dynamic and will be affected from the game evolution
    #TODO: it might be worth to integrate them as a param in the simulation function, only the name?

    #output variables:
    n_correct_per_episode = [0]*epochs
    n_wrong_per_episode = [0]*epochs
    n_timeout_per_episode = [0]*epochs
    n_max_attempt_per_episode = [0]*epochs
    game_performance_episode = [0]*epochs
    n_lev_0_no_feed_per_episode = [0]*epochs
    n_lev_1_no_feed_per_episode = [0]*epochs
    n_lev_2_no_feed_per_episode = [0]*epochs
    n_lev_3_no_feed_per_episode = [0]*epochs
    n_lev_4_no_feed_per_episode = [0]*epochs
    n_lev_0_with_feed_per_episode = [0]*epochs
    n_lev_1_with_feed_per_episode = [0]*epochs
    n_lev_2_with_feed_per_episode = [0]*epochs
    n_lev_3_with_feed_per_episode = [0]*epochs
    n_lev_4_with_feed_per_episode = [0]*epochs
    robot_assistance_per_episode = [0]*epochs

    for e in range(epochs):
        '''Simulation framework'''
        #counters
        task_evolution = 0
        attempt_counter = 0
        iter_counter = 0
        correct_move_counter = 0
        wrong_move_counter = 0
        timeout_counter = 0
        max_attempt_counter = 0
        robot_assistance_action = 0
        robot_feedback_action = 0

        dynamic_variables = {'attempt': attempt_counter_per_action,
                             'game_state': game_state_counter_per_action,
                             'robot_assistance': robot_assistance_per_action,
                             'robot_feedback': robot_feedback_per_action}

        while(task_evolution<task_complexity):
            #if then else are necessary to classify the task game state into beg, mid, end
            if task_evolution>=0 and task_evolution<=1:
                game_state_counter = 0
            elif task_evolution>=2 and task_evolution<=3:
                game_state_counter = 1
            else:
                game_state_counter = 2

            robot_vars_evidence = {     user_reactivity_name: user_reactivity_value,
                                        user_memory_name: user_memory_value,
                                        task_progress_name: game_state_counter,
                                        game_attempt_name: attempt_counter,
                                        }
            robot_actions_prob = bn_functions.infer_prob_from_state(robot_bn_model,
                                                                   infer_variable=robot_var_target,
                                                                   evidence_variables=robot_vars_evidence)

            robot_action = bn_functions.get_stochastic_action(robot_actions_prob.values)
            n_robot_assistance_feedback = Robot_Assistance_Feedback.counter.value
            if robot_action>=n_robot_assistance_feedback/2:
                robot_feedback_action = 1
                robot_assistance_action = n_robot_assistance_feedback-robot_action-1
                if robot_assistance_action == 0:
                    n_lev_0_no_feed_per_episode[e] += 1
                elif robot_assistance_action == 1:
                    n_lev_1_no_feed_per_episode[e] += 1
                elif robot_assistance_action == 2:
                    n_lev_2_no_feed_per_episode[e] += 1
                elif robot_assistance_action == 3:
                    n_lev_3_no_feed_per_episode[e] += 1
                else:
                    n_lev_4_no_feed_per_episode[e] += 1
            else:
                robot_feedback_action = 0
                robot_assistance_action = robot_action
                if robot_assistance_action == 0:
                    n_lev_0_with_feed_per_episode[e] += 1
                elif robot_assistance_action == 1:
                    n_lev_1_with_feed_per_episode[e] += 1
                elif robot_assistance_action == 2:
                    n_lev_2_with_feed_per_episode[e] += 1
                elif robot_assistance_action == 3:
                    n_lev_3_with_feed_per_episode[e] += 1
                else:
                    n_lev_4_with_feed_per_episode[e] += 1


            print("robot_assistance {}, attempt {}, game {}, robot_feedback {}".format(robot_assistance_action, attempt_counter, game_state_counter, robot_feedback_action))

            #compare the real user with the estimated Persona and returns a user action (0, 1, 2)
            if other_user_bn_model!=None:
                #return the user action in this state based on the user profile
                other_user_vars_evidence = {other_user_attention_name:other_user_attention_value,
                                            other_user_reactivity_name:other_user_reactivity_value,
                                            other_user_memory_name:other_user_memory_value,
                                            task_progress_name:game_state_counter,
                                            game_attempt_name:attempt_counter,
                                            robot_assistance_name:robot_assistance_action,
                                            robot_feedback_name:robot_feedback_action
                                            }
                user_actions_prob = bn_functions.infer_prob_from_state(other_user_bn_model,
                                                                       infer_variable=other_user_var_target,
                                                                       evidence_variables=other_user_vars_evidence)

            else:
                #return the user action in this state based on the Persona profile

                user_vars_evidence = {other_user_attention_name: user_attention_value,
                                            user_reactivity_name: user_reactivity_value,
                                            user_memory_name: user_memory_value,
                                            task_progress_name: game_state_counter,
                                            game_attempt_name: attempt_counter,
                                            robot_assistance_name: robot_assistance_action,
                                            robot_feedback_name: robot_feedback_action
                                            }
                query_res_prob = bn_functions.infer_prob_from_state(user_bn_model,
                                                                       infer_variable=user_var_target,
                                                                       evidence_variables=user_vars_evidence)

            #this is needed because we are querying the system with user_react_time and user_action output is 3x3
            query_res_prob_to_list = [query_res_prob.values[i][j]  for i in range(query_res_prob.values.shape[0]) for j in range (query_res_prob.values.shape[1])]
            query_action = bn_functions.get_stochastic_action(query_res_prob_to_list)
            user_action, user_react_time = bn_functions.interpret_user_output(query_action)
            #updates counters for plots
            robot_assistance_per_action[user_action][robot_assistance_action] += 1
            attempt_counter_per_action[user_action][attempt_counter] += 1
            game_state_counter_per_action[user_action][game_state_counter] += 1
            robot_feedback_per_action[user_action][robot_feedback_action] += 1

            #updates counters for simulation
            iter_counter += 1
            task_evolution, attempt_counter, correct_move_counter, \
            wrong_move_counter, timeout_counter, max_attempt_counter = compute_next_state(user_action,
                                                                        task_evolution, attempt_counter,
                                                                        correct_move_counter, wrong_move_counter,
                                                                        timeout_counter, max_attempt_counter)

        print("task_evolution {}, attempt_counter {}, timeout_counter {}".format(task_evolution, iter_counter, timeout_counter))
        print("robot_assistance_per_action {}".format(robot_assistance_per_action))
        print("attempt_counter_per_action {}".format(attempt_counter_per_action))
        print("game_state_counter_per_action {}".format(game_state_counter_per_action))
        print("robot_feedback_per_action {}".format(robot_feedback_per_action))
        print("iter {}, correct {}, wrong {}, timeout {}".format(iter_counter, correct_move_counter, wrong_move_counter, timeout_counter))
        print("correct_move {}, wrong_move {}, timeout {}".format(correct_move_counter, wrong_move_counter, timeout_counter))


        user_bn_model = bn_functions.update_cpds_tables(user_bn_model, dynamic_variables)
        #reset counter??
        robot_assistance_per_action = [[0 for i in range(Robot_Assistance.counter.value)] for j in
                                         range(User_Action.counter.value)]
        attempt_counter_per_action = [[0 for i in range(Attempt.counter.value)] for j in
                                      range(User_Action.counter.value)]
        game_state_counter_per_action = [[0 for i in range(Game_State.counter.value)] for j in
                                         range(User_Action.counter.value)]
        robot_feedback_per_action = [[0 for i in range(Robot_Feedback.counter.value)] for j in
                                     range(User_Action.counter.value)]

        #for plots
        n_correct_per_episode[e] = correct_move_counter
        n_wrong_per_episode[e] = wrong_move_counter
        n_timeout_per_episode[e] = timeout_counter
        n_max_attempt_per_episode[e] = max_attempt_counter
        game_performance_episode[e] = [n_correct_per_episode[e],
                                       n_wrong_per_episode[e],
                                       n_timeout_per_episode[e],
                                       n_max_attempt_per_episode[e]]
        robot_assistance_per_episode[e] = [n_lev_0_no_feed_per_episode[e],
        n_lev_1_no_feed_per_episode[e], n_lev_2_no_feed_per_episode[e],
        n_lev_3_no_feed_per_episode[e], n_lev_4_no_feed_per_episode[e],
        n_lev_0_with_feed_per_episode[e], n_lev_1_with_feed_per_episode[e],
        n_lev_2_with_feed_per_episode[e], n_lev_3_with_feed_per_episode[e],
                                           n_lev_4_with_feed_per_episode[e]
        ]

    return game_performance_episode, robot_assistance_per_episode



#############################################################################
#############################################################################
####################### RUN THE SIMULATION ##################################
#############################################################################
#############################################################################



#SIMULATION PARAMS
robot_assistance = [i for i in range(Robot_Assistance.counter.value)]
robot_feedback = [i for i in range(Robot_Feedback.counter.value)]
epochs = 40

#initialise the robot
robot_cpds = bnlearn.import_DAG('bn_robot_model/robot_model.bif')
#initialise memory, attention and reactivity varibles
persona_memory = 0; persona_attention = 0; persona_reactivity = 1;
persona_cpds = bnlearn.import_DAG('bn_persona_model/new_persona_model.bif')
#initialise memory, attention and reactivity varibles
real_user_memory = 2; real_user_attention = 2; real_user_reactivity = 2;
real_user_cpds = None#bnlearn.import_DAG('bn_other_user_model/user_model.bif')

game_performance_per_episode, robot_assistance_per_episode = simulation(user_bn_model=persona_cpds, user_var_target=['user_action', 'user_react_time'], user_memory_name="memory", user_memory_value=persona_memory,
                 user_attention_name="attention", user_attention_value=persona_attention,
            user_reactivity_name="reactivity", user_reactivity_value=persona_reactivity,
           task_progress_name="game_state", game_attempt_name="attempt",
                 robot_assistance_name="robot_assistance", robot_feedback_name="robot_feedback",

           robot_bn_model=robot_cpds, robot_var_target=["robot_assistance_feedback"],
           other_user_bn_model=real_user_cpds, other_user_var_target=['user_action'],
                 other_user_memory_name="memory", other_user_memory_value=real_user_memory,
           other_user_attention_name="attention", other_user_attention_value=real_user_attention,
           other_user_reactivity_name="reactivity", other_user_reactivity_value=real_user_reactivity,
           epochs=epochs, task_complexity=5)

plot_game_performance_path = ""
plot_robot_assistance_path = ""

if real_user_cpds != None:
    plot_game_performance_path = "game_performance_"+"_epoch_"+str(epochs)+"_real_user_memory_"+str(real_user_memory)+"_real_user_attention_"+str(real_user_attention)+"_real_user_reactivity_"+str(real_user_reactivity)+".jpg"
    plot_robot_assistance_path = "robot_assistance_"+"epoch_"+str(epochs)+"_real_user_memory_"+str(real_user_memory)+"_real_user_attention_"+str(real_user_attention)+"_real_user_reactivity_"+str(real_user_reactivity)+".jpg"
else:
    plot_game_performance_path = "game_performance_"+"epoch_" + str(epochs) + "_persona_memory_" + str(persona_memory) + "_persona_attention_" + str(persona_attention) + "_persona_reactivity_" + str(persona_reactivity) + ".jpg"
    plot_robot_assistance_path = "robot_assistance_"+"epoch_"+str(epochs)+"_persona_memory_"+str(persona_memory)+"_persona_attention_"+str(persona_attention)+"_persona_reactivity_"+str(persona_reactivity)+".jpg"

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

utils.plot2D_game_performance(full_path+plot_game_performance_path, epochs, game_performance_per_episode)
utils.plot2D_assistance(full_path+plot_robot_assistance_path, epochs, robot_assistance_per_episode)

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




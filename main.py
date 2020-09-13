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

def simulation(user_bn_model, user_vars_target_action, user_memory_name, user_memory_value, user_attention_name, user_attention_value,
               user_reactivity_name, user_reactivity_value,
               task_progress_name, game_attempt_name, robot_assistance_name, robot_feedback_name,
               robot_bn_model, robot_vars_action,
               other_user_bn_model, other_user_vars_target_action, other_user_memory_name, other_user_memory_value,
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
    CPD for each attempt 1 to 4 store the number of user_action (correct, wrong and timeout
    '''
    attempt_counter_per_action = [[0 for i in range(Attempt.counter.value)]  for j in range(User_Action.counter.value)]
    '''
    CPD for each game_state 0 to 2 store the number user_action (correct, wrong and timeout
    '''
    game_state_counter_per_action = [[0 for i in range(Game_State.counter.value)]  for j in range(User_Action.counter.value)]
    '''
    CPD for each robot feedback store the number of user_action (correct, wrong and timeout)
    '''
    robot_feedback_per_action = [[0 for i in range(Robot_Feedback.counter.value)] for j in range(User_Action.counter.value)]
    '''
    CPD for each robot assistance store the number of user_action (correct, wrong and timeout
    '''
    robot_assistance_per_action = [[0 for i in range(Robot_Assistance.counter.value)] for j in range(User_Action.counter.value)]

    '''
    CPD for each attempt 1 to 4 store the number of user_react_time (slow, normal and fast)
    '''
    attempt_counter_per_react_time = [[0 for i in range(Attempt.counter.value)] for j in range(User_Action.counter.value)]
    '''
    CPD for each game_state 0 to 2 store the number user_react_time (slow, normal and fast)
    '''
    game_state_counter_per_react_time = [[0 for i in range(Game_State.counter.value)] for j in
                                     range(User_Action.counter.value)]
    '''
    CPD for each robot feedback store the number of user_react_time (slow, normal and fast)
    '''
    robot_feedback_per_react_time = [[0 for i in range(Robot_Feedback.counter.value)] for j in
                                 range(User_Action.counter.value)]
    '''
    CPD for each robot assistance store the number of user_react_time (slow, normal and fast)
    '''
    robot_assistance_per_react_time = [[0 for i in range(Robot_Assistance.counter.value)] for j in
                                   range(User_Action.counter.value)]

    '''
    CPD for each game_state 0 to 2  the number of robot assistance
    '''
    game_state_counter_per_robot_assistance = [[0 for i in range(Game_State.counter.value)] for j in
                                   range(Robot_Assistance.counter.value)]
    '''
        CPD for each game_state 0 to 2  the number of robot assistance
    '''
    game_state_counter_per_robot_feedback = [[0 for i in range(Game_State.counter.value)] for j in
                                               range(Robot_Feedback.counter.value)]

    '''
    CPD for each attempt 1 to 4 store the number of robot_feedback
    '''
    attempt_counter_per_robot_assistance = [[0 for i in range(Attempt.counter.value)] for j in
                                   range(Robot_Assistance.counter.value)]
    '''
       CPD for each attempt 1 to 4 store the number of robot_feedback
       '''
    attempt_counter_per_robot_feedback = [[0 for i in range(Attempt.counter.value)] for j in
                                          range(Robot_Feedback.counter.value)]

    #these are the variables of the persona bn that are dynamic and will be affected from the game evolution
    #TODO: it might be worth to integrate them as a param in the simulation function, only the name?

    #output variables:
    n_correct_per_episode = [0]*epochs
    n_wrong_per_episode = [0]*epochs
    n_timeout_per_episode = [0]*epochs
    n_max_attempt_per_episode = [0]*epochs
    game_performance_episode = [0]*epochs
    n_assistance_lev_per_episode = [[0 for i in range(Robot_Assistance.counter.value)] for j in range(epochs)]
    n_feedback_per_episode = [[0 for i in range(Robot_Feedback.counter.value)] for j in range(epochs)]

    for e in range(epochs):
        '''Simulation framework'''
        #counters
        task_evolution_counter = 0
        attempt_counter = 0
        iter_counter = 0
        correct_move_counter = 0
        wrong_move_counter = 0
        timeout_counter = 0
        max_attempt_counter = 0
        robot_assistance_action_counter = 0
        robot_feedback_action_counter = 0

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

        while(task_evolution_counter<task_complexity):
            #if then else are necessary to classify the task game state into beg, mid, end
            if task_evolution_counter>=0 and task_evolution_counter<=1:
                game_state_counter = 0
            elif task_evolution_counter>=2 and task_evolution_counter<=3:
                game_state_counter = 1
            else:
                game_state_counter = 2

            robot_vars_evidence = {user_reactivity_name: user_reactivity_value,
                                   user_memory_name: user_memory_value,
                                   task_progress_name: game_state_counter,
                                   game_attempt_name: attempt_counter,
                                   }
            query_robot_action_prob = bn_functions.infer_prob_from_state(robot_bn_model,
                                                                   infer_variable=robot_vars_action,
                                                                   evidence_variables=robot_vars_evidence)
            # query_robot_feedback_prob = bn_functions.infer_prob_from_state(robot_bn_model,
            #                                                           infer_variable=robot_var_feedback_action,
            #                                                           evidence_variables=robot_vars_evidence)

            flatten_query_robot_prob, cols, rows = bn_functions.flat_action_probs(query_robot_action_prob)
            selected_robot_action = bn_functions.get_stochastic_action(flatten_query_robot_prob)
            #remember to pass the name of the variables that give us the right order to process them
            selected_robot_assistance_action, selected_robot_feedback_action = bn_functions.interpret_action_output(selected_robot_action, cols, rows, query_robot_action_prob.variables)
            n_assistance_lev_per_episode[e][selected_robot_assistance_action] += 1
            n_feedback_per_episode[e][selected_robot_feedback_action] += 1
            print("robot_assistance {}, attempt {}, game {}, robot_feedback {}".format(selected_robot_assistance_action, attempt_counter, game_state_counter, selected_robot_feedback_action))

            #compare the real user with the estimated Persona and returns a user action (0, 1, 2)
            if other_user_bn_model!=None:
                #return the user action in this state based on the user profile
                other_user_vars_evidence = {other_user_attention_name:other_user_attention_value,
                                            other_user_reactivity_name:other_user_reactivity_value,
                                            other_user_memory_name:other_user_memory_value,
                                            task_progress_name:game_state_counter,
                                            game_attempt_name:attempt_counter,
                                            robot_assistance_name:selected_robot_assistance_action,
                                            robot_feedback_name:selected_robot_feedback_action
                                            }
                query_user_action_prob = bn_functions.infer_prob_from_state(other_user_bn_model,
                                                                            infer_variable=other_user_vars_target_action,
                                                                            evidence_variables=other_user_vars_evidence)
            else:
                #return the user action in this state based on the Persona profile

                user_vars_evidence = {      user_attention_name: user_attention_value,
                                            user_reactivity_name: user_reactivity_value,
                                            user_memory_name: user_memory_value,
                                            task_progress_name: game_state_counter,
                                            game_attempt_name: attempt_counter,
                                            robot_assistance_name: selected_robot_assistance_action,
                                            robot_feedback_name: selected_robot_feedback_action
                                            }
                query_user_action_prob = bn_functions.infer_prob_from_state(user_bn_model,
                                                                       infer_variable=user_vars_target_action,
                                                                       evidence_variables=user_vars_evidence)

            flatten_query_user_prob, cols, rows = bn_functions.flat_action_probs(query_user_action_prob)
            selected_user_action = bn_functions.get_stochastic_action(flatten_query_user_prob)
            # remember to pass the name of the variables that give us the right order to process them
            selected_user_movement, selected_user_react_time = bn_functions.interpret_action_output(
                selected_user_action, cols, rows, query_user_action_prob.variables)

            #updates counters for user model
            robot_assistance_per_action[selected_user_movement][selected_robot_assistance_action] += 1
            attempt_counter_per_action[selected_user_movement][attempt_counter] += 1
            game_state_counter_per_action[selected_user_movement][game_state_counter] += 1
            robot_feedback_per_action[selected_user_movement][selected_robot_feedback_action] += 1

            robot_assistance_per_react_time[selected_user_react_time][selected_robot_assistance_action] += 1
            attempt_counter_per_react_time[selected_user_react_time][attempt_counter] += 1
            game_state_counter_per_react_time[selected_user_react_time][game_state_counter] += 1
            robot_feedback_per_react_time[selected_user_react_time][selected_robot_feedback_action] += 1

            game_state_counter_per_robot_assistance[selected_robot_assistance_action][game_state_counter] += 1
            attempt_counter_per_robot_assistance[selected_robot_assistance_action][attempt_counter] += 1
            game_state_counter_per_robot_feedback[selected_robot_feedback_action][game_state_counter] += 1
            attempt_counter_per_robot_feedback[selected_robot_feedback_action][attempt_counter] += 1

            #updates counters for simulation and compute the next state
            iter_counter += 1
            task_evolution_counter, attempt_counter, correct_move_counter, \
            wrong_move_counter, timeout_counter, max_attempt_counter = compute_next_state(selected_user_movement,
                                                                        task_evolution_counter, attempt_counter,
                                                                        correct_move_counter, wrong_move_counter,
                                                                        timeout_counter, max_attempt_counter)

        ####################################END of EPISODE#######################################
        print("task_evolution {}, attempt_counter {}, timeout_counter {}".format(task_evolution_counter, iter_counter, timeout_counter))
        print("robot_assistance_per_action {}".format(robot_assistance_per_action))
        print("attempt_counter_per_action {}".format(attempt_counter_per_action))
        print("game_state_counter_per_action {}".format(game_state_counter_per_action))
        print("robot_feedback_per_action {}".format(robot_feedback_per_action))
        print("iter {}, correct {}, wrong {}, timeout {}".format(iter_counter, correct_move_counter, wrong_move_counter, timeout_counter))
        print("correct_move {}, wrong_move {}, timeout {}".format(correct_move_counter, wrong_move_counter, timeout_counter))

        #update user model
        user_bn_model = bn_functions.update_cpds_tables(user_bn_model, user_action_dynamic_variables)
        user_bn_model = bn_functions.update_cpds_tables(user_bn_model, user_react_time_dynamic_variables)
        #update robot model
        robot_bn_model = bn_functions.update_cpds_tables(robot_bn_model, robot_assistance_dynamic_variables)
        robot_bn_model = bn_functions.update_cpds_tables(robot_bn_model, robot_feedback_dynamic_variables)

        #reset counter
        robot_assistance_per_action = [[0 for i in range(Robot_Assistance.counter.value)] for j in
                                         range(User_Action.counter.value)]
        robot_feedback_per_action = [[0 for i in range(Robot_Feedback.counter.value)] for j in
                                     range(User_Action.counter.value)]
        game_state_counter_per_action = [[0 for i in range(Game_State.counter.value)] for j in
                                         range(User_Action.counter.value)]
        attempt_counter_per_action = [[0 for i in range(Attempt.counter.value)] for j in
                                      range(User_Action.counter.value)]

        robot_assistance_per_react_time = [[0 for i in range(Robot_Assistance.counter.value)] for j in
                                         range(User_Action.counter.value)]
        robot_feedback_per_react_time = [[0 for i in range(Robot_Feedback.counter.value)] for j in
                                     range(User_Action.counter.value)]
        game_state_counter_per_react_time = [[0 for i in range(Game_State.counter.value)] for j in
                                         range(User_Action.counter.value)]
        attempt_counter_per_react_time = [[0 for i in range(Attempt.counter.value)] for j in
                                      range(User_Action.counter.value)]

        game_state_counter_per_robot_assistance = [[0 for i in range(Game_State.counter.value)] for j in
                                                   range(Robot_Assistance.counter.value)]
        game_state_counter_per_robot_feedback = [[0 for i in range(Game_State.counter.value)] for j in
                                                 range(Robot_Feedback.counter.value)]
        attempt_counter_per_robot_assistance = [[0 for i in range(Attempt.counter.value)] for j in
                                                range(Robot_Assistance.counter.value)]
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


    return game_performance_episode, n_assistance_lev_per_episode, n_feedback_per_episode



#############################################################################
#############################################################################
####################### RUN THE SIMULATION ##################################
#############################################################################
#############################################################################



#SIMULATION PARAMS
robot_assistance = [i for i in range(Robot_Assistance.counter.value)]
robot_feedback = [i for i in range(Robot_Feedback.counter.value)]
epochs = 10

#initialise the robot
robot_cpds = bnlearn.import_DAG('bn_robot_model/robot_model.bif')
#initialise memory, attention and reactivity varibles
persona_memory = 0; persona_attention = 0; persona_reactivity = 1;
persona_cpds = bnlearn.import_DAG('bn_persona_model/new_persona_model.bif')
#initialise memory, attention and reactivity varibles
real_user_memory = 2; real_user_attention = 2; real_user_reactivity = 2;
real_user_cpds = None#bnlearn.import_DAG('bn_other_user_model/user_model.bif')

game_performance_per_episode, robot_assistance_per_episode, robot_feedback_per_episode = \
            simulation(user_bn_model=persona_cpds, user_vars_target_action=['user_action', 'user_react_time'],
            user_memory_name="memory", user_memory_value=persona_memory,
            user_attention_name="attention", user_attention_value=persona_attention,
            user_reactivity_name="reactivity", user_reactivity_value=persona_reactivity,
            task_progress_name="game_state", game_attempt_name="attempt",
            robot_assistance_name="robot_assistance", robot_feedback_name="robot_feedback",
           robot_bn_model=robot_cpds, robot_vars_action=["robot_assistance", "robot_feedback"],
           other_user_bn_model=real_user_cpds, other_user_vars_target_action=['user_action', 'user_react_time'],
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




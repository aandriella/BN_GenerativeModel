import random
import bn_functions

def compute_next_state(user_action, task_evolution, attempt_counter, correct_move_counter,
                       wrong_move_counter, timeout_counter
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
    Return:
        the counters updated according to the user_action
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
        correct_move_counter += 1

    return task_evolution, attempt_counter, correct_move_counter, wrong_move_counter, timeout_counter

def get_user_action_prob():
    



def get_stochatic_action(actions_prob):
    '''
    Select one of the actions according to the actions_prob
    Args:
        actions_prob: the probability of the Persona based on the BN to make a correct move, wrong move, timeout
    Return:
        the id of the selected action
    N.B:
    '''
    action_id = None
    correct_action_from_BN = actions_prob[0]
    wrong_action_from_BN = actions_prob[1]
    timeout_action_from_BN = actions_prob[2]

    rnd_val = random.uniform(0,1)
    #if user_prob is lower than the correct action prob then is the correct one
    if rnd_val<=correct_action_from_BN:
        action_id = 0
    #if rnd is larger than the correct action prob and lower than wrong
    #  action prob then is the wrong one
    elif rnd_val>correct_action_from_BN \
        and rnd_val<(correct_action_from_BN+wrong_action_from_BN):
        action_id = 1
    #timeout
    else:
        action_id = 2
    return action_id

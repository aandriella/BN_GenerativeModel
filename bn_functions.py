import bnlearn
import numpy as np
import random
import os
import utils

'''Father class in which the basic functionalities to call the bnlearn library are developed
This class can be used to implement individual Persona simulator or just generative models starting from 
the bayesian network model
'''

def get_cpdf(dag_cpds, variable):
    '''
    This function returns given the bn model and the variable the cpds for variable
    :param dag_cpds: the bn model
    :param variable: the variable from which we want to get the probabiity
    Return:
        index: is the index of the variable in the model
        cpds_table[counter].values: values of the given variable
    '''
    cpds_table = (dag_cpds['model'].cpds[:])
    index = 0
    while (index < len(cpds_table)):
        if (cpds_table[index].variable == variable):
            return index, (cpds_table[index].values)
            break
        index += 1
    return None, None

def compute_prob(cpds_table):
    '''
    Given the counters generate the probability distributions
    Args:
        cpds_table: with counters
    Return:
         the probs for the cpds table
    '''

    def check_zero_occurrences(table):
        if sum(table) == 0:
                table = [1/len(table) for i in range(len(table))]
        return table
    '''
    This function checks if any 
    '''

    for val in range(len(cpds_table)):
            cpds_table[val] = list(map(lambda x: x / (sum(cpds_table[val])+0.00001), cpds_table[val]))
            cpds_table[val] = check_zero_occurrences(cpds_table[val])
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

def update_cpds_tables(bn_model, variables_tables):
    '''
    This function updates the bn model with the variables_tables provided in input
    Args:
        variables_table: it is a dict {'variable': val} where var is the counter of the occurrence of a varible
        bn_model: the cpds tables of the model to update
    Return:
        the cpds tables updated with the new counters
    '''
    # transform counters into probabilities
    for key, val in variables_tables.items():
        # get the values of a given table
        index, cpds_table = get_cpdf(bn_model, key)
        # from the counters recreate the tables
        cpds_table_from_counter = compute_prob(val)
        updated_prob = average_prob(
            np.transpose(cpds_table),
            cpds_table_from_counter)
        bn_model['model'].cpds[index].values = np.transpose(updated_prob)

    return bn_model

def get_dynamic_variables(evidence_variables_name, evidence_variables_value):
    '''
    This func returns a dict of the form name:value and it defines the "evidences"
     that will be used to query the BN
    Args:
        :evidence_variables_name: the name of the variable
        :evidence_variables_value: the value of the given variable
    Return:
         a dict of the form name:value
    '''
    if len(evidence_variables_name)!=len(evidence_variables_value):
        assert "The variables name numbers is different from the variables value"
    else:
        dynamic_variables = {evidence_variables_name[i]:evidence_variables_value[i] for i in range(len(evidence_variables_name))}
        return dynamic_variables

def infer_prob_from_state(user_bn_model, infer_variable, evidence_variables):
    '''
    Given the model, the variable to infer, and the evidences returns the distribution prob for that variable
    Args:
        user_bn_model:
        infer_variable:
        evidence_variables:
        :
    Returns:
        the probability distribution for varibale_to_infer
    '''
    dist_prob = bnlearn.bnlearn.inference.fit(user_bn_model, variables=infer_variable,
                                                              evidence=evidence_variables)
    return dist_prob

def get_stochastic_action(actions_distr_prob):
    '''
    Select one of the actions according to the actions_prob
    Args:
        actions_prob: the probability of the Persona based on the BN to make a correct move, wrong move, timeout
    Return:
        the id of the selected action
    N.B:
    '''
    def compute_distance(values, target):
        '''
        Return the index of the most closest value in values to target
        Args:
            target: the target value
            values: a list of values from 0 to 1a
        Return:
             return the index of the value closer to target
        '''
        min_dist = 1
        index = 0
        for i in range(len(values)):
            if abs(target-values[i])<min_dist:
                min_value = values[i]
                min_dist = abs(target-values[i])
                #check the delta to assign the correct bin
                if (target - min_value) > 0 and index != len(values) - 1:
                    index = i + 1
                else:
                    index = i
        return index


    actions_distr_prob_scaled = [0]*len(actions_distr_prob)
    accum = 0
    for i in range(len(actions_distr_prob)):
        accum += actions_distr_prob[i]
        actions_distr_prob_scaled[i] = accum

    rnd_val = random.uniform(0, 1)
    action_id = compute_distance(actions_distr_prob_scaled, rnd_val)

    return action_id

def flat_action_probs(action_probs):
    flat_array_user_action_prob = None
    column = 0
    if len(action_probs.values.shape)==2:
        flat_array_user_action_prob = [action_probs.values[j][i] for j in range(action_probs.values.shape[0]) for i in range(action_probs.values.shape[1])]
        column = action_probs.values.shape[0]
        row = action_probs.values.shape[1]
    else:
        assert "Did you forget to add the additional target, only one has been detected"
        flat_array_user_action_prob = action_probs.values
        column = 1
        row = 0
    return flat_array_user_action_prob, column, row



def interpret_action_output(action_id, col, row, targets):
    '''
       Given the id of the action selected from the probabilistic inference model and the target variables
        return the action (user act + react time) or (robot ass and robot feedback)
       Args
           action_id 1d array of probs
           col: #col of array action id
           row: #row of array action id
           targets the targets we aim to evaluate
       Return:
           user_action
           user_react_time
       '''
    #N.B it assumes that the query is performed passing as first argument robot_assistance
    # and as a second robot_feedback
    robot_assistance = 0
    robot_feedback = 0
    user_action = 0
    user_react_time = 0

    if targets[0] == 'user_action':
        user_action = int(action_id / row)
        user_react_time = int(action_id % row)
        print("user_action ", user_action, ' user_react ', user_react_time)
        return user_action, user_react_time
    elif targets[1] == 'user_action':
        user_action = int(action_id % row)
        user_react_time = int(action_id / row)
        print("user_action ", user_action, ' user_react ', user_react_time)
        return user_action, user_react_time
    elif targets[0] == "robot_assistance":
        robot_assistance = int(action_id / row)
        robot_feedback = int(action_id % row)
        print("robot_ass ", robot_assistance, 'robot_feed ', robot_feedback)
        return robot_assistance, robot_feedback
    elif targets[1] == "robot_assistance":
        robot_assistance = int(action_id % row)
        robot_feedback = int(action_id / row)
        print("robot_feed ", robot_assistance, 'robot_ass ', robot_feedback)
        return robot_assistance, robot_feedback

def update_episodes_batch(bn_model_user_action, bn_model_user_react_time,
                          bn_model_agent_assistance, bn_model_agent_feedback,
                          folder_filename, with_caregiver=True):
    bn_belief_user_action_file = "bn_belief_user_action.pkl"
    bn_belief_user_react_time_file = "bn_belief_user_react_time.pkl"
    bn_belief_agent_assistance_file = ""; bn_belief_agent_feedback_file = "";
    if with_caregiver:
        bn_belief_agent_assistance_file = "bn_belief_caregiver_assistive_action.pkl"
        bn_belief_agent_feedback_file = "bn_belief_caregiver_feedback_action.pkl"
    elif not with_caregiver:
        bn_belief_agent_assistance_file = "bn_belief_robot_assistive_action.pkl"
        bn_belief_agent_feedback_file = "bn_belief_robot_feedback_action.pkl"


    #check if the folder is empty
    dir = os.listdir(path=folder_filename)
    if dir==[]:
        assert "Folder is empty"
        return
    else:
        dir_list = next(os.walk(folder_filename))[1]
        for sub_folder in dir_list:
            #read the files in it (we already know their name)
            bn_belief_user_action = utils.read_user_statistics_from_pickle(folder_filename+"/"+sub_folder+"/"+bn_belief_user_action_file)
            bn_belief_user_react_time = utils.read_user_statistics_from_pickle(folder_filename+"/"+sub_folder+"/"+bn_belief_user_react_time_file)
            bn_belief_agent_assistance = utils.read_user_statistics_from_pickle(folder_filename+"/"+sub_folder+"/"+bn_belief_agent_assistance_file)
            if bn_model_agent_feedback != None:
                bn_belief_agent_feedback = utils.read_user_statistics_from_pickle(folder_filename+"/"+sub_folder+"/"+bn_belief_agent_feedback_file)
                bn_model_agent_feedback = update_cpds_tables(bn_model=bn_model_agent_feedback,
                                                             variables_tables=bn_belief_agent_feedback)

            bn_model_user_action = update_cpds_tables(bn_model=bn_model_user_action,
                                                      variables_tables=bn_belief_user_action)
            bn_model_user_react_time = update_cpds_tables(bn_model=bn_model_user_react_time,
                                                      variables_tables=bn_belief_user_react_time)
            bn_model_agent_assistance = update_cpds_tables(bn_model=bn_model_agent_assistance,
                                                      variables_tables=bn_belief_agent_assistance)

    #return the 4 models
    return bn_model_user_action, bn_model_user_react_time, bn_model_agent_assistance, bn_model_agent_feedback

# bn_model_caregiver_assistance = bnlearn.import_DAG('bn_agent_model/agent_assistive_model.bif')
# bn_model_caregiver_feedback = bnlearn.import_DAG('bn_agent_model/agent_feedback_model.bif')
# bn_model_user_action = bnlearn.import_DAG('bn_persona_model/user_action_model.bif')
# bn_model_user_react_time = bnlearn.import_DAG('bn_persona_model/user_react_time_model.bif')
# update_episodes_batch(bn_model_user_action, bn_model_user_react_time, bn_model_caregiver_assistance,
#                       bn_model_caregiver_feedback, folder_filename="/home/pal/carf_ws/src/carf/caregiver_in_the_loop/log/1/0",
#                       with_caregiver=True, with_feedback=False)


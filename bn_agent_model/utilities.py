import random
import bn_functions


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

def infer_prob(user_bn_model, variable_to_infer, evidence_vars_name, evidence_vars_value):
    '''
    Given the model, the variable to infer, and the evidences returns the distribution prob for that variable
    Args:
        user_bn_model:
        variable_to_infer:
        evidence_vars_name:
        evidence_vars_value:
    Returns:
        the probability distribution for varibale_to_infer
    '''
    evidence = get_dynamic_variables(evidence_vars_name, evidence_vars_value)
    dist_prob = bn_functions.get_inference_from_state(user_bn_model,
                                                              variables=variable_to_infer,
                                                              evidence=evidence)
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
            values: a list of values from 0 to 1
        Return:
             return the index of the value closer to target
        '''
        min_dist = 1
        index = 0
        for i in range(len(values)):
            if abs(target-values[i])<min_dist:
                min_dist = abs(target-values[i])
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


actions_prob_distr =  [0.32, 0.105, 0.035, 0.035, 0.005, 0.36,  0.065, 0.035, 0.035, 0.005]
action_index = get_stochastic_action(actions_prob_distr)
print(action_index)
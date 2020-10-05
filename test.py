import bnlearn as bn
import pandas as pd


def import_data_from_csv(csv_filename, dag_filename):
    print("/************************************************************/")
    print("Init model")
    DAG = bn.import_DAG(dag_filename)
    df_caregiver = bn.sampling(DAG, n= 10000)
    print("/************************************************************/")
    print("real_user Model")
    DAG_ = bn.import_DAG(dag_filename, CPD=False)
    df_real_user = pd.read_csv(csv_filename)
    DAG_real_user = bn.parameter_learning.fit(DAG_, df_real_user, methodtype='bayes')
    df_real_user = bn.sampling(DAG_real_user, n=10000)
    print("/************************************************************/")
    print("Shared knowledge")
    DAG_ = bn.import_DAG(dag_filename, CPD=False)
    shared_knowledge = [df_real_user, df_caregiver]
    conc_shared_knowledge = pd.concat(shared_knowledge)
    DAG_shared = bn.parameter_learning.fit(DAG_, conc_shared_knowledge)
    df_conc_shared_knowledge = bn.sampling(DAG_shared, n=10000)
    return DAG_shared



import_data_from_csv(csv_filename='bn_persona_model/cognitive_game.csv', dag_filename='bn_persona_model/persona_model_test.bif')
# DAG = bn.import_DAG('bn_persona_model/persona_model_test.bif')
# G = bn.plot(DAG)
# q1 = bn.inference.fit(DAG, variables=[ 'user_action'], evidence={
#                                                                 'game_state': 0,
#                                                                 'attempt':0,
#                                                                 'agent_feedback':1,
#                                                                 'memory': 0,
#                                                                 'reactivity':0,
#                                                                 'agent_assistance':0,
#
# })
# df = pd.read_csv('bn_persona_model/cognitive_game.csv')
# df = bn.sampling(DAG, n=10000)
# #model_sl = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# #bn.plot(model_sl, pos=G['pos'])
# DAG_update = bn.parameter_learning.fit(DAG, df)
# n_game_state = 3
# n_attempt = 4
# n_aas = 6
# n_af = 2

# for gs in range(n_game_state):
#     for att in range(n_attempt):
#         for aas in range(n_aas):
#             for af in range(n_af):
#                 q1 = bn.inference.fit(DAG_update, variables=[ 'user_action'], evidence={
#                                                                 'game_state': gs,
#                                                                 'attempt':att,
#                                                                 'agent_feedback':af,
#                                                                 'user_memory':0,
#                                                                 'user_reactivity':0,
#                                                                 'agent_assistance':aas})
#                 print("GS:", gs, " ATT:", att, " AA", aas, " AF", af)
#
# df.head()
# DAG = bn.import_DAG('bn_persona_model/persona_model_test.bif', CPD=False)
# bn.plot(DAG)
# DAG_update = bn.parameter_learning.fit(DAG, df)
# DAG_true = bn.import_DAG('bn_persona_model/persona_model_test.bif', CPD=True)
# q1 = bn.inference.fit(DAG_update, variables=['user_action'], evidence={
#                                                                 'game_state': 0,
#                                                                 'attempt':2,
#                                                                 'agent_feedback':0,
# })
# print("BEFORE")
# print(q1.values)
# df = bn.sampling(DAG_update, n=1000)
# DAG_update = bn.parameter_learning.fit(DAG_update, df)
# q1 = bn.inference.fit(DAG_update, variables=['user_action'], evidence={
#                                                                 'game_state': 0,
#                                                                 'attempt':2,
#                                                                 'agent_feedback':0,
# })
# print("AFTER")
# print(q1.values)

#df = bn.sampling(DAG, n=1000, verbose=2)
#model = bn.structure_learning.fit(df)
#G = bn.plot(model)
#DAGnew = bn.parameter_learning.fit(model, df, methodtype="bayes")
#bn.print_CPD(DAGnew)
# q1 = bn.inference.fit(DAG, variables=['user_action'], evidence={
#                                                                 'game_state': 0,
#                                                                 'attempt':1,
#                                                                 'agent_feedback':1,
# })
# print(q1.values)

# robot_assistance = [0, 1, 2, 3, 4, 5]
# attempt_t0 = [0, 1, 2, 3]
# game_state_t0 = [0, 1, 2]
# attempt_t1 = [0]
# game_state_t1 = [0, 1, 2]
#
# query_result = [[[0 for j in range(len(attempt_t0))] for i in range(len(robot_assistance))] for k in range(len(game_state_t0))]
# for k in range(len(game_state_t0)):
#     for i in range(len(robot_assistance)):
#         for j in range(len(attempt_t0)-1):
#             if j == 0:
#                 query = bn.inference.fit(DAG, variables=['user_action'],
#                 evidence={'game_state_t0': k,
#                 'attempt_t0': j,
#                 'agent_assistance': i,
#                 'game_state_t1': k,
#                 'attempt_t1': j})
#                 query_result[k][i][j] = query.values
#             else:
#                 query = bn.inference.fit(DAG, variables=['user_action'],
#                                              evidence={'game_state_t0': k,
#                                                        'attempt_t0': j,
#                                                        'agent_assistance': i,
#                                                        'game_state_t1': k,
#                                                        'attempt_t1': j + 1})
#                 query_result[k][i][j] = query.values
# for k in range(len(game_state_t0)):
#     for i in range(len(robot_assistance)):
#         for j in range(len(attempt_t0)):
#             if j == 0:
#                 print("game_state:",k, "attempt_from:", j," attempt_to:",j, " robot_ass:",i, " prob:", query_result[k][i][j])
#             else:
#                 print("game_state:", k, "attempt_from:", j, " attempt_to:", j+1, " robot_ass:", i, " prob:",
#                       query_result[k][i][j])

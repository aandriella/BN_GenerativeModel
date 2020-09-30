import bnlearn as bn

DAG = bn.import_DAG('bn_persona_model/persona_model_test.bif')

#df = bn.sampling(DAG, n=1000, verbose=2)
#model = bn.structure_learning.fit(df)
#G = bn.plot(model)
#DAGnew = bn.parameter_learning.fit(model, df, methodtype="bayes")
#bn.print_CPD(DAGnew)
q1 = bn.inference.fit(DAG, variables=['user_action'], evidence={
                                                                'game_state_t0': 0,
                                                                'attempt_t0':1,
                                                                'game_state_t1': 0,
                                                                'attempt_t1':2,
                                                                'agent_assistance':0,
})
print(q1.values)

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

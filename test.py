import bnlearn as bn

DAG = bn.import_DAG('bn_persona_model/persona_model_test.bif')

#df = bn.sampling(DAG, n=1000, verbose=2)
#model = bn.structure_learning.fit(df)
#G = bn.plot(model)
#DAGnew = bn.parameter_learning.fit(model, df, methodtype="bayes")
#bn.print_CPD(DAGnew)
q1 = bn.inference.fit(DAG, variables=['user_action'], evidence={
                                                                'game_state_t0': 1,
                                                                'attempt_t0':0,
                                                                'robot_assistance':5,
                                                                'game_state_t1': 1,
                                                                'attempt_t0':1,


})

print(q1.variables)
print(q1.values)

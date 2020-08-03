import bnlearn
import numpy as np

#df = bnlearn.import_example()
model = bnlearn.import_DAG('persona_model_3.bif')
#q_1 = bnlearn.inference.fit(model, variables=['Rain'], evidence={'Cloudy':1, 'Sprinkler':0, 'Wet_Grass':1})
#q_2 = bnlearn.inference.fit(model, variables=['Rain'], evidence={'Cloudy':1})
df = bnlearn.sampling(model, n=1000)
print(df)
q_1 = bnlearn.inference.fit(model, variables=['user_action'], evidence={'robot_assistance':0,
                                                                        'attempt':2,
                                                                         'game_state':0,
                                                                         'robot_feedback':0,
                                                                         'memory':1,
                                                                         'reactivity':1,
                                                                         'attention':1
                                                                        })
print(q_1)
# update = np.arange(9).reshape(3, 3)
# model['model'].cpds[4].values[0][0] = update
# print(model['model'].cpds[4].values[0][0])
#print("model 0")
#print(model["model"].cpds[0].values)
#print("model 1")
#print(model["model"].cpds[1].values)
#print("model 2")
#print(model["model"].cpds[2].values)
#print("model 3")
#print(model["model"].cpds[3].values)
#print("model 4")
#print(model["model"].cpds[4].values)
#print("model 5")
#print(model["model"].cpds[5].values)
#print("model 6")
#print(model["model"].cpds[6].values)
#print("model 7")
#print(model["model"].cpds[7].values)

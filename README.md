### Bayesian Network to generate patient and therapist (human or robotic) models ### 
##### The two BN models are initialised by using the data collected from the human-therapist patient interactions and the therapist expertise in the form of questionnaire. The models are then refined session after session when the robot interacts with the patient. Therefore more accurate models are estimated.



#### Package:
- **bn_model_template** folder contains the models skeleton of the two models in bn_learn format
- **bn_functions.py** and bn_variables.py are auxiliary classes for the bn_learn lib
- **utils.py** auxiliary script for plotting the results of the simulation
- **simulation.py** given the two models it generates the simulated interactions between the therapist and the patient.
It returns a sequence of episodes defines as <current_state, assistive_action, next_state> that will be used as input for learning the reward function specific for that patient and therefore the degrees of assistance they need.


#### NB: in order to build the BNs that models the therapist and the patient we used the [bnlearn](https://pypi.org/project/bnlearn/) library

from __future__ import (unicode_literals, print_function, absolute_import,
                        division)


from ema_workbench import (Model, MultiprocessingEvaluator, Policy,
                           Scenario)

from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging
import time
from problem_formulation import get_model_for_problem_formulation
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import prim

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    dike_model, planning_steps = get_model_for_problem_formulation(2)

    # Build a user-defined scenario and policy:
    reference_values = {'Bmax': 175, 'Brate': 1.5, 'pfail': 0.5,
                        'ID flood wave shape': 4, 'planning steps': 2}
    reference_values.update({'discount rate {}'.format(n): 3.5 for n in planning_steps})
    scen1 = {}

    for key in dike_model.uncertainties:
        name_split = key.name.split('_')

        if len(name_split) == 1:
            scen1.update({key.name: reference_values[key.name]})

        else:
            scen1.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario('reference', **scen1)

    # no dike increase, no warning, none of the rfr
    zero_policy = {'DaysToThreat': 0}
    zero_policy.update({'DikeIncrease {}'.format(n): 0 for n in planning_steps})
    zero_policy.update({'RfR {}'.format(n): 0 for n in planning_steps})
    pol0 = {}

    for key in dike_model.levers:
        s1, s2 = key.name.split('_')
        pol0.update({key.name: zero_policy[s2]})

    policy0 = Policy('Policy 0', **pol0)

    # Call random scenarios or policies:
#    n_scenarios = 5
#    scenarios = sample_uncertainties(dike_model, 50)
#    n_policies = 10

    # single run    
    # start = time.time()
    # dike_model.run_model(ref_scenario, policy0)
    # end = time.time()
    # print(end - start)
    # results = dike_model.outcomes_output


#Series run RUN with reference Case
    # Define number of scenarios or reference scenario
    # scenarios = ref_scenario # #SPECIFY NUMBER OF SCENARIOS OR REFERENCE CASE
    # n_policies = 5
    # results = perform_experiments(dike_model, scenarios, n_policies)
    # experiments, outcomes = results    
   
# print (outcomes)
# Multiprocessing


    with MultiprocessingEvaluator(dike_model) as evaluator:
        results = evaluator.perform_experiments(scenarios=10, policies=[policy0,policy1,policy2,policy3]
                                                uncertainty_sampling='sobol')

    experiments, outcomes = results
    policies = experiments['policy']
    
    data = pd.DataFrame.from_dict(outcomes)
    data['policy'] = policies
    
    sns.pairplot(data, hue='policy',  vars=outcomes.keys(), )
    plt.show()

#%%



    

#%% Save results
# this code can be used to save the experiments done, 
# uncomment it and press ctrl+enter to do this
'''
 # If scenarios is not a number specifying the number of scenarios, but a Policy object, assume ref scenario is run
    if isinstance(scenarios, int):
        n_scenarios = scenarios
    else:
        n_scenarios = 1
 
    initials= "FD"
    fn = 'results/{} scenarios {} policies_{}.tar.gz'.format(n_scenarios, n_policies, initials)
    from ema_workbench import save_results
    save_results(results, fn)
'''
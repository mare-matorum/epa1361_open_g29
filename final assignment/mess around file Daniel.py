# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:50:38 2021

@author: danie
"""

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
from ema_workbench import RealParameter, ScalarOutcome
import numpy as np
from ema_workbench.analysis import prim
from ema_workbench import Policy

#%%

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



#%%   
    ###this defines policy 0
    # no dike increase, no warning, none of the rfr
    zero_policy = {'DaysToThreat': 0}
    zero_policy.update({'DikeIncrease {}'.format(n): 0 for n in planning_steps})
    zero_policy.update({'RfR {}'.format(n): 0 for n in planning_steps})
    
    pol0 = {}
    for key in dike_model.levers:
        s1, s2 = key.name.split('_')
        pol0.update({key.name: zero_policy[s2]})

    new_policy = {}
        
#%%
    ###this defines policy 1 (maximally heightening all dikes)
    heightening_policy = {'DaysToThreat': 2}
    # heightening_policy.update({'A.1_DikeIncrease 2'.format(n): 2 for n in planning_steps})
    heightening_policy.update({'DikeIncrease {}'.format(n): 5 for n in planning_steps})
    heightening_policy.update({'RfR {}'.format(n): 0 for n in planning_steps})
    
    pol1 = {}
    for key in dike_model.levers:
        s1, s2 = key.name.split('_')
        pol1.update({key.name: heightening_policy[s2]})

#%%
    ###this defines policy 2 (only early warning system)
    warning_policy = {'DaysToThreat': 2}
    warning_policy.update({'DikeIncrease {}'.format(n): 0 for n in planning_steps})
    warning_policy.update({'RfR {}'.format(n): 1 for n in planning_steps})
   
    pol2 = {}
    for key in dike_model.levers:
        s1, s2 = key.name.split('_')
        pol2.update({key.name: warning_policy[s2]})

    policy0 = Policy('Nothing', **pol0)
    policy1 = Policy("only heightening", **pol1)
    policy2 = Policy("only room for river", **pol2)  

    policies = [policy0, policy1, policy2]
    
#%%
    # with MultiprocessingEvaluator(dike_model) as evaluator:
    #     results = evaluator.perform_experiments(scenarios=50, 
    #                                                 policies=
    #                                                 [policy0,policy1,
    #                                                   policy2,policy3])

    with MultiprocessingEvaluator(dike_model) as evaluator:
        results = evaluator.perform_experiments(
                                                scenarios = 2000, 
                                                policies = policies,
                                                )
    experiments, outcomes = results
    
    
    policies = experiments['policy'] 
    data = pd.DataFrame.from_dict(outcomes)
    data['policy'] = policies
    sns.pairplot(data, hue='policy',  vars=outcomes.keys())
    plt.show()
#%%
    #This section computes total costs
    dfoutcomes = pd.DataFrame(outcomes)
    dfoutcomes['Total costs'] = dfoutcomes.sum(axis=1)
    dfresults = pd.concat([experiments,dfoutcomes], axis=1)

    # This section can export the results to an excelsheet.    
    death_threshold = 0.0001
    y = outcomes['Expected Number of Deaths'] < 0.0001
    dfresults['Death Risk Condition < {}'.format(str(death_threshold))] = y


    
    #this section saves the results to excel if allowed
    to_excel = False
    if to_excel == True:
        timestamp = time.strftime("%m.%d-%H%M%S")    
        dfresults.to_excel(r'.\results{}.xlsx'.format(timestamp), index = False)

    to_tar = True
    if to_tar == True:
        from ema_workbench import save_results
        save_results(results, 'HyperCube_open_exploration_iterations.tar.gz')



#%% Prim
    # This section sets the conditions that are acceptable for our analysis
    cleaned_experiments = experiments.drop(labels=[l.name for l in dike_model.levers], axis=1)
    y = (dfoutcomes['Expected Number of Deaths'] < death_threshold) & (dfoutcomes['Total costs'] > 0)
    #y.value_counts()

    prim_alg = prim.Prim(cleaned_experiments,y, threshold=0.8)
    box1 = prim_alg.find_box()
    box1.show_tradeoff()
    plt.show()

    box1.inspect(2)
    box1.inspect(2, style='graph')
    plt.show()


#%% Save results
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


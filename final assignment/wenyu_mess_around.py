#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 17:34:55 2021

@author: wenyuc
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

from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS
from ema_workbench.analysis import feature_scoring
from ema_workbench.analysis.scenario_discovery_util import RuleInductionType
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from SALib.analyze import sobol


from ema_workbench.analysis import prim

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
    
#%%

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
    scenarios = ref_scenario # #SPECIFY NUMBER OF SCENARIOS OR REFERENCE CASE
    n_policies = 5
    results = perform_experiments(dike_model, scenarios, n_policies)
    experiments, outcomes = results    
   
# print (outcomes)

    # data = pd.DataFrame.from_dict(outcomes)
    # data['policy'] = policies
    
    # sns.pairplot(data, hue='policy',  vars=outcomes.keys(), )
    # plt.show()

#%%

# Explore the behaviour of the system in the absence of any policy 

    n_scenarios= 1000
    policy = policy0

    with MultiprocessingEvaluator(dike_model) as evaluator:
        results = evaluator.perform_experiments(n_scenarios, policy)
            
        experiments, outcomes = results
        policies = experiments['policy']
    

    sns.pairplot(pd.DataFrame.from_dict(outcomes))
    plt.show()

# Explore the effect of dike failure probability at each location to expected annual damage and deaths

    cleaned = experiments.loc[:, [u.name for u in dike_model.uncertainties]]
    print(cleaned)

    cleaned['Expected Number of Deaths'] = outcomes['Expected Number of Deaths']
    cleaned['Expected Annual Damage'] = outcomes['Expected Annual Damage']


    fig, ax = plt.subplots(figsize=(6,6))

    m = ax.scatter(cleaned['A.1_pfail'], cleaned['Expected Annual Damage'], c=cleaned['Expected Number of Deaths'])
    ax.set_xlabel('A.1 dike failure probability')
    ax.set_ylabel('Expected Annual Damage')
    fig.colorbar(m)

    plt.show()


#%%

# Sobol analysis: To investigate which model inputs add variance to the model output (expected 
# annual damage and expected number of deaths) under no policy 

    n_scenarios= 50
    policy = policy0
    
    with MultiprocessingEvaluator(dike_model) as evaluator:
        results = evaluator.perform_experiments(n_scenarios, policy, uncertainty_sampling=SOBOL)
        
    experiments, outcomes = results

    problem = get_SALib_problem(dike_model.uncertainties)
    y = outcomes 
    
    sobol_indices1 = sobol.analyze(problem, outcomes['Expected Annual Damage'])
    sobol_indices2 = sobol.analyze(problem, outcomes['Expected Number of Deaths'])

    sobol_indices_list = [sobol_indices1, sobol_indices2]
    
    for n in range(0,2):
        sobol_stats = {key:sobol_indices_list[n][key] for key in ['ST', 'ST_conf', 'S1','S1_conf']}
        sobol_stats = pd.DataFrame(sobol_stats, index=problem['names'])
        sobol_stats.sort_values(by='ST', ascending=False)
        
        sns.set_style('white')
        fig, ax = plt.subplots(1)
        
        indices = sobol_stats[['S1','ST']]
        err = sobol_stats[['S1_conf','ST_conf']]
        
        indices.plot.bar(yerr=err.values.T,ax=ax)
        fig.set_size_inches(8,6)
        fig.subplots_adjust(bottom=0.3)
        plt.ylim([0,1])
        plt.grid()
        
    plt.show()






#%%



    heightening_policy = {'DaysToThreat': 0}
    heightening_policy.update({'DikeIncrease {}'.format(n): 0 for n in planning_steps})
    heightening_policy.update({'RfR {}'.format(n): 0 for n in planning_steps})
    
    pol1 = {}
    for key in dike_model.levers:
        s1, s2 = key.name.split('_')
        pol1.update({key.name: heightening_policy[s2]})
        pol1.update({'A.1_DikeIncrease 0'.format(n): 10 for n in planning_steps})
        
    policy1 = Policy("only heightening", **pol1)
    
    n_scenarios= 50
    
    with MultiprocessingEvaluator(dike_model) as evaluator:
        results = evaluator.perform_experiments(ref_scenario, policies=policy1, levers_sampling=LHS)
                                                
        experiments, outcomes = results
        
    sns.pairplot(pd.DataFrame.from_dict(outcomes))
    plt.show()
  

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
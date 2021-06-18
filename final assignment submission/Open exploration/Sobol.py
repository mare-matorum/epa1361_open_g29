# Explore the behaviour of the system in the absence of any policy 
# Sobol analysis: To investigate which model inputs add variance to the model output (expected 
# annual damage and expected number of deaths) under no policy 

from __future__ import (unicode_literals, print_function, absolute_import,
                        division)
from ema_workbench import (Model, MultiprocessingEvaluator, Policy, IntegerParameter,
                           Scenario)
import numpy as np
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
 

# Perform Sobil analysis
    n_scenarios= 1000
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
    

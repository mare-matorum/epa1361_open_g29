# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:28:36 2021

@author: danie
"""

#%%
if __name__ == '__main__':    
#this defines policy 0
    # no dike increase, no warning, none of the rfr
    zero_policy = {'DaysToThreat': 0}
    zero_policy.update({'DikeIncrease {}'.format(n): 0 for n in planning_steps})
    zero_policy.update({'RfR {}'.format(n): 0 for n in planning_steps})
    
    pol0 = {}
    for key in dike_model.levers:
        s1, s2 = key.name.split('_')
        pol0.update({key.name: zero_policy[s2]})


#this defines policy 1 (maximally heightening dikes)
    heightening_policy = {'DaysToThreat': 0}
    heightening_policy.update({'DikeIncrease {}'.format(n): 10 for n in planning_steps})
    heightening_policy.update({'RfR {}'.format(n): 0 for n in planning_steps})
    #heightening_policy.update({'DikeIncrease {}'.format(n): 10 for n in planning_steps})
    #heightening_policy.update({'RfR {}'.format(n): 0 or n in planning_steps}) 
    
    pol1 = {}
    for key in dike_model.levers:
        s1, s2 = key.name.split('_')
        pol1.update({key.name: heightening_policy[s2]})

#this defines policy 2 (only early warning system)


#this defines policy 3 (a maximized approach)
    

    policy0 = Policy('Policy 0', **pol0)
    policy1 = Policy("only heightening", **pol1)
    #policy2 = Policy("only evacuating", **pol2)
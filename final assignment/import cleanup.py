
import pandas as pd
from ema_workbench import load_results

results = pd.read_csv("Optimizationresults 100K.csv") 

#Sum the different RfR values to identify the one with 0 for the our municiaplites. identify intresting solutions
results ["sum rfr"] = (results ['0_RfR 0'] + 
                       results ['0_RfR 1'] +
                       results ['0_RfR 2'] +
                       results ['1_RfR 0'] + 
                       results ['1_RfR 1'] +
                       results ['1_RfR 2'] +
                       results ['2_RfR 0'] + 
                       results ['2_RfR 1'] +
                       results ['2_RfR 2'])




results ["sum deaths"] = (
                        results ['A.1_Expected Number of Deaths'] + 
                        results ['A.2_Expected Number of Deaths'] +
                        results ['A.3_Expected Number of Deaths'] +
                        results ['A.4_Expected Number of Deaths'] + 
                        results ['A.5_Expected Number of Deaths']
                       )
                     
# This code selects preferences based on our preferences
#no RFR costs for dikes
#no deaths in dike 1
#0.00148 for dike 2
#0.001 for dike 3 
int_solutions = (
                (results["sum rfr"] <= 0) & 
                (results['A.1_Expected Number of Deaths'] <= 0.0) &
                (results['A.2_Expected Number of Deaths'] <= 0.00148) &
                (results['A.3_Expected Number of Deaths'] <= 0.001)
                )
int_solutions.value_counts()
results['Satisfying'] = int_solutions
satisfying_solutions = results[results['okay']==True]


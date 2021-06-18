# README of Group 29
This file explains the organization of the final submission folder of Group 29.

The files are organized in folders named after the boxes in the modelling strategy.

## Folder structure
The *MORDM folder* includes the script that is used to search policies using MOEAs. Furthermore, this folder includes all result files, most importantly:
- "Optimizationresults 100K.csv": Containing optimization results from MOEA application
- "SECOND candidate solutions MORDM 1000scenarios.tar.gz": Containing the scenarios performed for the five candidate solutions (1000 experiments for each of the 5 candidate solutions = 5000 scenarios)

- The *Robustness folder* contains the Jupyter notebook used to calculate robustness based on the experiments performed for five candidate solutions.
- The *Open exploration* folder contains files used for open exploration to gain insights into the model and possible tradeoffs.
- The *Scenario discovery* folder contains files used to investigate the outcomes of MORDM.

*Important: All scripts except for open exploration rely on results from MORDM. In case you would like to reproduce results, make sure to adjust the import paths in the scripts accordingly so that they point to the MORDM folder.*


# MOGPHH
This repository includes the code related to the paper:
Surrogate-assisted automatic evolving of dispatching rules for multi-objective dynamic job shop scheduling using genetic programming
https://doi.org/10.1016/j.eswa.2022.118194

# Abstract
Dispatching rules are simple but efficient heuristics to solve multi-objective job shop scheduling problems, particularly useful to face the challenges of dynamic shop environments. A promising method to automatically evolve non-dominated rules represents multi-objective genetic programming based hyper-heuristic (MO-GP-HH). The aim of such methods is to approximate the Pareto front of non-dominated dispatching rules as good as possible in order to provide a sufficient set of efficient solutions from which the decision maker can select the most preferred one. However, one of the main drawbacks of existing approaches is the computational demanding simulation-based fitness evaluation of the evolving rules. To efficiently allocate the computational budget, surrogate models can be employed to approximate the fitness. Two possible ways, that estimate the fitness either based on a simplified problem or based on samples of fully evaluated individuals making use of machine learning techniques are investigated in this paper. Several representatives of both categories are first examined with regard to their selection accuracy and execution time. Furthermore, we developed a surrogate-assisted MO-GP-HH framework, incorporating a pre-selection task in the NSGA-II algorithm. The most promising candidates are consequently implemented in the framework. Using a dynamic job shop scenario, the two proposed algorithms are compared to the original one without using surrogates. With the aim to minimize the mean flowtime and maximum tardiness, experimental results demonstrate that the proposed algorithms outperform the former. Making use of surrogates leads to a reduction in computational costs of up to 70%. Another interesting finding shows that the enhanced ability to identify duplicates based on the phenotypic characterization of individuals is particularly helpful in increasing diversity within a population. This study illustrates the positive effect of this mechanism on the exploration of the entire Pareto front.

# Keywords
`Multi-objective optimization` `Surrogates` `Hyper-heuristic` `Genetic programming` `Dynamic job shop scheduling` `Machine learning`

# Glossary 
- `main.py` - Main file where the loop of runs is implemented (1 to 30)
- `GPHH.py` -Implementation of the genetic programming (definition of terminals, functions, and other settings)
- `deap/algorithms.py` - Includes the MO-GP-HH algorithms (loop over all generations):
  - `GPHH_experiment1` - Algorithm for the first experiment to measure the selection accuracy and execution time
  - `GPHH_experiment2_WS` - Algorithm for the second experiment without using surrogates
  - `GPHH_experiment2_SR` - Algorithm for the second experiment using SR as a surrogate
  - `GPHH_experiment2_RF` - Algorithm for the second experiment using RF as a surrogate
  - `GPHH_experiment2_RF_duplicate` - Algorithm for the second experiment using RF as a surrogate and eliminating duplicates based on phenotypic characterization
  - `GPHH_experiment2_DT_duplicates` - Algorithm for the second experiment using DT as a surrogate and eliminating duplicates based on phenotypic characterization
  - `GPHH_experiment2_NB_duplicates` - Algorithm for the second experiment using NB as a surrogate and eliminating duplicates based on phenotypic characterization
- `phenotypic_generator.py` - Functions needed to compute the phenotypic characterization of the individuals
- `surrogates.py` - Functions for the different surrogates
- `simulation.py` - Discrete event simulation to evaluate the fitness of individuals
- `testing.py` - Evaluating the final population of each run on an independent set of simulation instances
- `experiments/Experiment1.py` - Experiment 1: selection accuracy and execution time: evaluating the generated data and creating the visualizations
- `experiments/Experiment2.py` - Experiment 2: convergence curve and boxplots: evaluating the generated data and creating the visualization
- `experiments/Experiment3.py` - Experiment 3: populations: evaluating the final populations of each run and creating the visualizations
- `experiments/Experiment4.py` - Experiment 4: testing: evaluating the testing performance and creating the visualizations

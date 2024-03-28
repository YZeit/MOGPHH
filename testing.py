import multiprocessing
import numpy as np
from simulation import simulation
import operator
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import pygmo as pg
import pandas as pd
import time
import os

def div(left, right):
    if right == 0:
        return 1
    else:
        return left / right

def ifte(condition, return_if_true, return_if_not_true):
    if condition >= 0:
        argument = return_if_true
    else:
        argument = return_if_not_true
    return argument

# define the functions and terminals
pset = gp.PrimitiveSet("MAIN", 13)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(ifte, 3)

# rename the terminals
# job/operation
pset.renameArguments(ARG0='PT')
pset.renameArguments(ARG1='RT')
pset.renameArguments(ARG2='RPT')
pset.renameArguments(ARG3='RNO')
pset.renameArguments(ARG4='DD')
pset.renameArguments(ARG5='RTO')
pset.renameArguments(ARG6='PTN')
pset.renameArguments(ARG7='SL')
pset.renameArguments(ARG8='WT')
# work center
pset.renameArguments(ARG9='APTQ')
pset.renameArguments(ARG10='NJQ')
# global
pset.renameArguments(ARG11='WINQ')
pset.renameArguments(ARG12='CT')

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, phenotypic=None)

# set some GP parameters
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def testing(input):
    # Transform the tree expression in a callable function
    print(input)
    func = toolbox.compile(expr=input)
    makespan, max_tardiness, waiting_time, mean_flowtime = [], [], [], []
    random_seed = [255, 49, 201, 126, 50, 133, 118, 13, 249, 93, 60, 82, 221, 23, 196, 45, 157, 5, 171, 298, 122,
                   67, 280, 132, 138, 142, 38, 4, 199, 279, 80, 79, 273, 145, 274, 216, 83, 98, 193, 278, 155, 227,
                   258, 56, 43, 48, 73, 81, 63, 29]
    random_seed = [2, 4, 6]
    for s in random_seed:
        current_max_tardiness, current_mean_flowtime = \
            simulation(number_machines=10, number_jobs=2500, warm_up=500, func=func, random_seed=s,
                       due_date_tightness=4, utilization=0.80, missing_operation=True)
        # makespan.append(current_makespan)
        max_tardiness.append(current_max_tardiness)
        # waiting_time.append(current_waiting_time)
        mean_flowtime.append(current_mean_flowtime)
    # current_makespan = np.mean(makespan)
    current_max_tardiness = np.mean(max_tardiness)
    # current_waiting_time = np.mean(waiting_time)
    current_mean_flowtime = np.mean(mean_flowtime)
    return current_mean_flowtime, current_max_tardiness

def main():
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # load the population file
    link = 'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/WS/run_5/population.xlsx'
    population_WS = pd.read_excel(link, header=0, index_col=0)

    # load the population file
    link = 'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/RF/run_1/population.xlsx'
    population_RF = pd.read_excel(link, header=0, index_col=0)

    final_population_WS = population_WS['Individuals Generation 50']
    final_population_WS = final_population_WS.dropna()
    final_population_WS = list(final_population_WS)

    final_population_RF = population_RF['Individuals Generation 50']
    final_population_RF = final_population_RF.dropna()
    final_population_RF = list(final_population_RF)

    mean_flowtime_list = []
    max_tardiness_list = []

    '''
    for i in final_population_WS:
        start = time.time()
        mean_flowtime_current_ind, max_tardiness_current_ind = testing(i)
        mean_flowtime_list.append(mean_flowtime_current_ind)
        max_tardiness_list.append(max_tardiness_current_ind)
        end = time.time()
        print(final_population_WS.index(i))
        print(mean_flowtime_current_ind)
        print(max_tardiness_current_ind)
        print(end-start)
    '''

    #mean_flowtime_list, max_tardiness_list = toolbox.map(testing, final_population_WS)
    result = toolbox.map(testing, final_population_WS)
    result = list(map(list, zip(*result)))
    mean_flowtime_list = result[0]
    max_tardiness_list = result[1]
    print(mean_flowtime_list)
    print(max_tardiness_list)

    testing_performance = {'mean flowtime': mean_flowtime_list, 'max tardiness': max_tardiness_list}
    testing_performance_df = pd.DataFrame(testing_performance)
    testing_performance_df.to_excel('D:/PycharmProjects/08_EMO_hyper_heuristic/Results/WS/run_2/testing.xlsx')


if __name__ == '__main__':
    main()




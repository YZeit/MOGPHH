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
import os
from phenotypic_generator import decision_situations_generator, ranking_vector_generator, decision_vector_generator, ranking_vector_generator_ref

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

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, phenotypic=None)

# set some GP parameters
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def simpleEvalgenSeed(input):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=input[0])
    random_seed_current_gen = input[1]
    current_max_tardiness, current_mean_flowtime = simulation(number_machines=10, number_jobs=2500, warm_up=500,
                                                                               func=func, random_seed=random_seed_current_gen,
                                                                               due_date_tightness=4, utilization=0.80, missing_operation=True)

    return current_mean_flowtime, current_max_tardiness

def simpleEvalfixSeed(input):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=input)
    random_seed_current_gen = 41
    current_max_tardiness, current_mean_flowtime = simulation(number_machines=10, number_jobs=2500, warm_up=500,
                                                                               func=func, random_seed=random_seed_current_gen,
                                                                               due_date_tightness=4, utilization=0.80, missing_operation=True)

    return current_mean_flowtime, current_max_tardiness

def fullEval(input):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=input)
    makespan, max_tardiness, waiting_time, mean_flowtime = [], [], [], []
    #random_seed = [4, 15, 384, 199, 260, 56, 79, 154, 117, 29]
    random_seed = [4, 15, 384, 199, 260]
    for s in random_seed:
        current_max_tardiness, current_mean_flowtime = \
            simulation(number_machines=10, number_jobs=2500, warm_up=500, func=func, random_seed=s,
                       due_date_tightness=4, utilization=0.80, missing_operation=True)
        #makespan.append(current_makespan)
        max_tardiness.append(current_max_tardiness)
        #waiting_time.append(current_waiting_time)
        mean_flowtime.append(current_mean_flowtime)
    #current_makespan = np.mean(makespan)
    current_max_tardiness = np.mean(max_tardiness)
    #current_waiting_time = np.mean(waiting_time)
    current_mean_flowtime = np.mean(mean_flowtime)
    return current_mean_flowtime, current_max_tardiness

def singlerep(input):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=input[0])
    random_seed_current_gen = input[1]
    current_max_tardiness, current_mean_flowtime = simulation(number_machines=10, number_jobs=2500, warm_up=500,
                                                                               func=func, random_seed=random_seed_current_gen,
                                                                               due_date_tightness=4, utilization=0.80, missing_operation=True)

    return current_mean_flowtime, current_max_tardiness

def singlerepshort(input):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=input[0])
    random_seed_current_gen = input[1]
    current_max_tardiness, current_mean_flowtime = simulation(number_machines=10, number_jobs=500, warm_up=100,
                                                                               func=func, random_seed=random_seed_current_gen,
                                                                               due_date_tightness=4, utilization=0.80, missing_operation=True)

    return current_mean_flowtime, current_max_tardiness

def halfshopEval(input):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=input[0])
    random_seed = input[1]
    current_max_tardiness, current_mean_flowtime = \
        simulation(number_machines=5, number_jobs=500, warm_up = 100,  func=func, random_seed=random_seed,
                   due_date_tightness=4, utilization=0.80, missing_operation=True)
    return current_mean_flowtime, current_max_tardiness

def test(input):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=input)
    makespan, max_tardiness, waiting_time, mean_flowtime = [], [], [], []
    random_seed = [10, 31, 44, 2, 344, 12, 455, 234, 103, 20]
    for s in random_seed:
        current_max_tardiness, current_mean_flowtime = \
            simulation(number_machines=10, number_jobs=2500, func=func, random_seed=s,
                       due_date_tightness=4, utilization=0.80, missing_operation=True)
        #makespan.append(current_makespan)
        max_tardiness.append(current_max_tardiness)
        #waiting_time.append(current_waiting_time)
        mean_flowtime.append(current_mean_flowtime)
    #current_makespan = np.mean(makespan)
    current_max_tardiness = np.mean(max_tardiness)
    #current_waiting_time = np.mean(waiting_time)
    current_mean_flowtime = np.mean(mean_flowtime)
    return current_mean_flowtime, current_max_tardiness

def decision_vector(input):
    func = toolbox.compile(expr=input[0])
    decision_situations = input[1]
    ranking_vector_ref = input[2]
    ranking_vector = ranking_vector_generator(individual_func=func, decision_situations=decision_situations)
    decision_vector_current_ind = decision_vector_generator(ranking_vector_ref=ranking_vector_ref, ranking_vector=ranking_vector)
    return decision_vector_current_ind


# initialize GP and set some parameter
toolbox.register("evaluate_singleRep", singlerep)
toolbox.register("evaluate_singleRepShort", singlerepshort)
toolbox.register("evaluate_halfshop", halfshopEval)
#toolbox.register("evaluate", simpleEvalfixSeed)
toolbox.register("full_evaluate", fullEval)
toolbox.register("decision_vector", decision_vector)
#toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

def main(run):
    #random.seed(318)

    # Enable multiprocessing using all CPU cores
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # define population and hall of fame (size of the best kept solutions found during GP run)
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)

    #history = tools.History() #todo: mal anschauen -> hier: https://www.programcreek.com/python/example/90740/deap.algorithms.eaSimple

    # define statistics for the GP run to be measured
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean, axis=0)
    mstats.register("std", np.std, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    # generate decision situations for the phenotypic characterization
    decision_situations = decision_situations_generator()
    # generate ref ranking vector
    ranking_vector_ref = ranking_vector_generator_ref(individual_func=0, decision_situations=decision_situations)

    # get population and logbook from the whole GP run
    #pop, log, selection_accuracy, selection_accuracy_simple, rank_correlation, time_dict = algorithms.GPHH_experiment1(pop, toolbox, 0.9, 0.1, 10, decision_situations, ranking_vector_ref,n =2, n_offspring=0.5, stats=mstats,
    #                           halloffame=hof, verbose=True)
    pop, log, time_dict, population_dict, fitness_dict, fitness_evaluations = algorithms.GPHH_experiment2_SR(pop, toolbox, 0.9, 0.1, 50, decision_situations, ranking_vector_ref,n =2, n_offspring=0.5, stats=mstats,
                               halloffame=hof, verbose=True)
    #pop, log = algorithms.eaSimple_new(pop, toolbox, 0.9, 0.1, 10, stats=mstats,
    #                                                       halloffame=hof, verbose=True)

    #decision_vectors_df = pd.DataFrame(decision_vectors)
    #decision_vectors_df.to_csv('decision_vectors.csv')

    # define the path where the results are supposed to be saved
    path = "D:/PycharmProjects/08_EMO_hyper_heuristic/Results/run_{a}".format(a=run)
    # create the new folder for each run
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


    pop_df = pd.DataFrame([[str(i), i.fitness] for i in pop])
    pop_df.to_excel(path+'/final_population.xlsx')


    #selection_accuracy_df = pd.DataFrame(selection_accuracy)
    #selection_accuracy_df.to_excel(path+'/selection_accuracy.xlsx')

    #selection_accuracy_simple_df = pd.DataFrame(selection_accuracy_simple)
    #selection_accuracy_simple_df.to_excel(path+'/selection_accuracy_simple.xlsx')

    #rank_correlation_df = pd.DataFrame(rank_correlation)
    #rank_correlation_df.to_excel(path+'/rank_correlation.xlsx')

    time_dict_df = pd.DataFrame(time_dict)
    time_dict_df.to_excel(path+'/time.xlsx')

    fitness_evaluations_df = pd.DataFrame.from_dict(fitness_evaluations, orient='index')
    #fitness_evaluations_df = fitness_evaluations_df.transpose()
    print(fitness_evaluations_df)
    fitness_evaluations_df.to_excel(path+'/fitness_evaluations.xlsx')

    population_dict_df = pd.DataFrame.from_dict(population_dict, orient='index')
    population_dict_df = population_dict_df.transpose()
    print(population_dict_df)
    population_dict_df.to_excel(path+'/population.xlsx')

    fitness_dict_df = pd.DataFrame.from_dict(fitness_dict, orient='index')
    fitness_dict_df = fitness_dict_df.transpose()
    print(fitness_dict_df)
    fitness_dict_df.to_excel(path+'/fitness.xlsx')

    # extract statistics:
    avgFitnessValues  = log.chapters['fitness'].select("avg")
    minFitnessValues = log.chapters['fitness'].select("min")
    maxFitnessValues = log.chapters['fitness'].select("max")
    stdFitnessValues = log.chapters['fitness'].select("std")
    nb_generation = log.select("gen")
    nevals = log.select('nevals')

    # transform statistics into numpy arrays
    minFitnessValues = np.array(minFitnessValues)
    maxFitnessValues = np.array(maxFitnessValues)
    avgFitnessValues = np.array(avgFitnessValues)
    stdFitnessValues = np.array(stdFitnessValues)
    nb_generation = np.array(nb_generation)

    # load best solution of the GP run
    #best_solution = hof.items[0]
    #best_fitness = best_solution.fitness.values[0]

    # perform a full evaluation on the whole population of the last generation
    #f1 = []
    #f2 = []
    #for p in pop:
    #    f1_current_p, f2_current_p = test(p)
    #    f1.append(f1_current_p)
    #    f2.append(f2_current_p)

    #df_solution = pd.DataFrame(
    #    {'f1': f1,
    #     'f2': f2,
    #     })

    #print(df_solution)

    #hyp = pg.hypervolume(df_solution[['f1','f2']].values)
    #hypervolume = hyp.compute([10000, 60000])
    #print(hypervolume)
    #hypervolume = 1
    #return nb_generation, avgFitnessValues, minFitnessValues, maxFitnessValues, stdFitnessValues



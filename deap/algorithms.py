#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every
evolutionary algorithm may vary infinitely. Most of the algorithms in this
module use operators registered in the toolbox. Generally, the keyword used are
:meth:`mate` for crossover, :meth:`mutate` for mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.

You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import random
import numpy as np
from . import tools
import time
from surrogates import DT_train, KNN_train, MLP_train, SVM_train, NB_train, LR_train, RF_train, predict, evaluate_selection_accuracy, evaluate_rank_correlation
from math import ceil

def remove_duplicates(population):
    new_population = []
    temp_list = []
    for ind in population:
        if str(ind) not in temp_list:
            new_population.append(ind)
            temp_list.append(str(ind))
    return new_population


def remove_duplicates_phenotypic(population):
    new_population = []
    temp_list = []
    for ind in population:
        if ind.phenotypic not in temp_list:
            new_population.append(ind)
            temp_list.append(ind.phenotypic)
    return new_population


def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]


    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            offspring[i - 1].phenotypic = None
            offspring[i].phenotypic = None

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            offspring[i].phenotypic = None
    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def GPHH(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref, n, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)
    '''
    # Evaluate the individuals with an invalid fitness using full evaluation
    invalid_ind = np.array([[ind, rand_value] for ind in population if not ind.fitness.valid], dtype=object)
    invalid_ind_1 = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind_1, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
'''
    # Update random seed
    rand_value = random.randint(1,300)

    # Initialize selection accuracy
    selection_accuracy_list = []
    selection_accuracy_list_simple = []

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = np.array([[ind, rand_value] for ind in population if not ind.fitness.valid], dtype=object)
        invalid_ind_1 = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind_1, fitnesses):
            ind.fitness.values = fit

        # generate the phenotypic characterization of the individuals
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population], dtype=object)
        invalid_ind_1 = [ind for ind in population]
        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = np.array(decision_matrix)
        fitness_matrix = [ind.fitness.values for ind in population]

        # Update the surrogate models (train the predictors)
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)

        # produce offsprings
        population_intermediate_to_add = []
        for i in range(n):
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate_to_add += offspring

        # initialize the intermediate population
        population_intermediate = population + population_intermediate_to_add


        # Evaluate the individuals with an invalid fitness
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate if not ind.fitness.valid], dtype=object)
        invalid_ind_1 = [ind for ind in population_intermediate if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind_1, fitnesses):
            ind.fitness.values = fit

        # Assign a number to each individual in the intermediate population
        for i in range(len(population_intermediate)):
            population_intermediate[i].number = i

        # generate the phenotypic characterization of the individuals in the intermediate population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate], dtype=object)
        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = np.array(decision_matrix)

        # Create a duplicate of the intermediate population for the prediction
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness
        prediction_KNN = predict(KNN_model, decision_matrix)
        prediction_MLP = predict(MLP_model, decision_matrix)
        prediction_DT = predict(DT_model, decision_matrix)
        prediction_SVM = predict(SVM_model, decision_matrix)

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit

        # Select the next generation individuals based on the evaluation
        pop_size = len(population)
        population = toolbox.select(population_intermediate, pop_size)

        # Select the next generation individuals based on the estimation (just for evaluation purposes, no real selection is made)
        population_predicted_KNN = toolbox.select(population_intermediate_predicted_KNN, pop_size)
        population_predicted_MLP = toolbox.select(population_intermediate_predicted_MLP, pop_size)
        population_predicted_DT = toolbox.select(population_intermediate_predicted_DT, pop_size)
        population_predicted_SVM = toolbox.select(population_intermediate_predicted_SVM, pop_size)

        # compare both selections and calculate the selection accuracy
        ranking = []
        ranking_predicted_KNN = []
        ranking_predicted_MLP = []
        ranking_predicted_DT = []
        ranking_predicted_SVM = []
        for i in population:
            ranking.append(i.number)
        for i in population_predicted_KNN:
            ranking_predicted_KNN.append(i.number)
        for i in population_predicted_MLP:
            ranking_predicted_MLP.append(i.number)
        for i in population_predicted_DT:
            ranking_predicted_DT.append(i.number)
        for i in population_predicted_SVM:
            ranking_predicted_SVM.append(i.number)
        print(ranking)
        print(ranking_predicted_KNN)
        selection_accuracy_KNN, selection_accuracy_KNN_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_KNN,
                                                         pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        selection_accuracy_MLP, selection_accuracy_MLP_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_MLP,
                                                         pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        selection_accuracy_DT, selection_accuracy_DT_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_DT,
                                                         pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        selection_accuracy_SVM, selection_accuracy_SVM_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_SVM,
                                                         pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))

        selection_accuracy_list.append([selection_accuracy_KNN, selection_accuracy_MLP, selection_accuracy_DT, selection_accuracy_SVM])
        selection_accuracy_list_simple.append([selection_accuracy_KNN_simple, selection_accuracy_MLP_simple, selection_accuracy_DT_simple, selection_accuracy_SVM_simple])

        print('selection accuracy ')
        print(f'KNN: {selection_accuracy_list}')
        print('selection accuracy simple')
        print(f'KNN: {selection_accuracy_list_simple}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Update random seed
        rand_value = random.randint(1, 300)

    return population, logbook, selection_accuracy_list, selection_accuracy_list_simple


def GPHH_new(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref, n, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)
    '''
    # Evaluate the individuals with an invalid fitness using full evaluation
    invalid_ind = np.array([[ind, rand_value] for ind in population if not ind.fitness.valid], dtype=object)
    invalid_ind_1 = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind_1, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
'''
    # Update random seed
    rand_value = random.randint(1,300)

    # Initialize selection accuracy
    selection_accuracy_list = []
    selection_accuracy_list_simple = []

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # produce offsprings
        offspring = varAnd(population, toolbox, cxpb, mutpb)
        population_1 = population + offspring

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = np.array([[ind, rand_value] for ind in population_1 if not ind.fitness.valid], dtype=object)
        invalid_ind_1 = [ind for ind in population_1 if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind_1, fitnesses):
            ind.fitness.values = fit

        # generate the phenotypic characterization of the individuals
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_1], dtype=object)
        invalid_ind_1 = [ind for ind in population_1]
        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = np.array(decision_matrix)
        fitness_matrix = [ind.fitness.values for ind in population_1]

        # Update the surrogate models (train the predictors)
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)

        # Select the next generation individuals based on the evaluation
        pop_size = len(population)
        population = toolbox.select(population_1, pop_size)

        # produce offsprings
        population_intermediate_to_add = []
        for i in range(n):
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate_to_add += offspring

        # initialize the intermediate population
        population_intermediate = population + population_intermediate_to_add


        # Evaluate the individuals with an invalid fitness
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate if not ind.fitness.valid], dtype=object)
        invalid_ind_1 = [ind for ind in population_intermediate if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind_1, fitnesses):
            ind.fitness.values = fit

        # Assign a number to each individual in the intermediate population
        for i in range(len(population_intermediate)):
            population_intermediate[i].number = i

        # generate the phenotypic characterization of the individuals in the intermediate population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate], dtype=object)
        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = np.array(decision_matrix)

        # Create a duplicate of the intermediate population for the prediction
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness
        prediction_KNN = predict(KNN_model, decision_matrix)
        prediction_MLP = predict(MLP_model, decision_matrix)
        prediction_DT = predict(DT_model, decision_matrix)
        prediction_SVM = predict(SVM_model, decision_matrix)

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit

        # Select the next generation individuals based on the evaluation
        pop_size = len(population)
        population = toolbox.select(population_intermediate, pop_size)

        # Select the next generation individuals based on the estimation (just for evaluation purposes, no real selection is made)
        population_predicted_KNN = toolbox.select(population_intermediate_predicted_KNN, pop_size)
        population_predicted_MLP = toolbox.select(population_intermediate_predicted_MLP, pop_size)
        population_predicted_DT = toolbox.select(population_intermediate_predicted_DT, pop_size)
        population_predicted_SVM = toolbox.select(population_intermediate_predicted_SVM, pop_size)

        # compare both selections and calculate the selection accuracy
        ranking = []
        ranking_predicted_KNN = []
        ranking_predicted_MLP = []
        ranking_predicted_DT = []
        ranking_predicted_SVM = []
        for i in population:
            ranking.append(i.number)
        for i in population_predicted_KNN:
            ranking_predicted_KNN.append(i.number)
        for i in population_predicted_MLP:
            ranking_predicted_MLP.append(i.number)
        for i in population_predicted_DT:
            ranking_predicted_DT.append(i.number)
        for i in population_predicted_SVM:
            ranking_predicted_SVM.append(i.number)
        print(ranking)
        print(ranking_predicted_KNN)
        selection_accuracy_KNN, selection_accuracy_KNN_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_KNN,
                                                         pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        selection_accuracy_MLP, selection_accuracy_MLP_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_MLP,
                                                         pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        selection_accuracy_DT, selection_accuracy_DT_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_DT,
                                                         pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        selection_accuracy_SVM, selection_accuracy_SVM_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_SVM,
                                                         pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))

        selection_accuracy_list.append([selection_accuracy_KNN, selection_accuracy_MLP, selection_accuracy_DT, selection_accuracy_SVM])
        selection_accuracy_list_simple.append([selection_accuracy_KNN_simple, selection_accuracy_MLP_simple, selection_accuracy_DT_simple, selection_accuracy_SVM_simple])

        print('selection accuracy ')
        print(f'KNN: {selection_accuracy_list}')
        print('selection accuracy simple')
        print(f'KNN: {selection_accuracy_list_simple}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Update random seed
        rand_value = random.randint(1, 300)

    return population, logbook, selection_accuracy_list, selection_accuracy_list_simple

# GPHH for testing the selection accuracy (using the latest 1000 fully evaluated individuals for training of surrogates)
def GPHH_experiment1(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref,n , n_offspring, stats=None,
             halloffame=None, verbose=__debug__):


    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)
    '''
    # Evaluate the individuals with an invalid fitness using full evaluation
    invalid_ind = np.array([[ind, rand_value] for ind in population if not ind.fitness.valid], dtype=object)
    invalid_ind_1 = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind_1, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
'''

    # initialize the lists for measuring the time
    KNN_time_train = []
    MLP_time_train = []
    DT_time_train = []
    SVM_time_train = []
    NB_time_train = []
    LR_time_train = []
    RF_time_train = []

    KNN_time_predict = []
    MLP_time_predict = []
    DT_time_predict = []
    SVM_time_predict = []
    NB_time_predict = []
    LR_time_predict = []
    RF_time_predict = []

    SR_time_predict = []
    SR_short_time_predict = []
    HS_time_predict = []

    phenotypic_time = []
    phenotypic_time_1 = []

    full_evaluation = []

    # Update random seed
    rand_value = random.randint(1,300)

    # Initialize selection accuracy
    selection_accuracy_list = []
    selection_accuracy_list_simple = []
    rank_correlation_list = []

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population)/2)
    pop_size_offspring = int(len(population))
    #print(f'Initial population:')
    #for i in population:
    #    print(str(i))
    # Begin the generational process
    for gen in range(1, ngen + 1):
        print(f'Generation: {gen}')
        population = remove_duplicates(population=population)
        #start = time.time()
        print(f'population size: {len(population)}')
        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # add the full evaluated individuals to the training set
        pop_train += invalid_ind
        if len(pop_train) > 1000:
            pop_train = pop_train[-1000:]
        print(f'population size of the training population: {len(pop_train)}')
        #print("\n")
        #print(f'Training population:')
        #for i in pop_train:
        #    print(str(i))
        #print("\n")
        # generate the phenotypic characterization of the individuals
        start = time.time()
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in pop_train], dtype=object)
        invalid_ind_1 = [ind for ind in pop_train]
        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = np.array(decision_matrix)
        fitness_matrix = [ind.fitness.values for ind in pop_train]
        end = time.time()
        time_needed = end - start
        phenotypic_time.append(time_needed)


        # Update the surrogate models (train the predictors)
        start = time.time()
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end-start
        KNN_time_train.append(time_needed)

        start = time.time()
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        MLP_time_train.append(time_needed)

        start = time.time()
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        DT_time_train.append(time_needed)

        start = time.time()
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        SVM_time_train.append(time_needed)

        start = time.time()
        NB_model = NB_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        NB_time_train.append(time_needed)

        start = time.time()
        LR_model = LR_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        LR_time_train.append(time_needed)

        start = time.time()
        RF_model = RF_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        RF_time_train.append(time_needed)

        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        #print("\n")
        #print(f'Selected population:')
        #for i in population:
        #    print(str(i))
        #print("\n")
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = len(population)*n
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]

        #for i in range(n+1):
        #    offspring = varAnd(population, toolbox, cxpb, mutpb)
        #    population_intermediate += offspring
        #    #print("\n")

        #print("\n")
        #print(f'Intermediate population:')
        #for i in population_intermediate:
        #    print(str(i))
        #print("\n")

        # remove duplicates
        #population_intermediate = remove_duplicates(population=population_intermediate)
        # remove already evaluated individuals (duplicates to te original population after selection)
        #population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]


        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:(len(population)*n)]

        print(f'population size of intermediate population: {len(population_intermediate)}')

        # Evaluate the individuals with an invalid fitness using full evaluation (only for comparison purpose to calculate the accuracy)
        start = time.time()
        invalid_ind = [ind for ind in population_intermediate if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        end = time.time()
        time_needed = end - start
        full_evaluation.append(time_needed)

        print(f'evaluated individuals of intermediate population: {len(invalid_ind)}')
        # add the full evaluated individual to the training population
        pop_train += invalid_ind

        # Assign a number to each individual in the intermediate population
        for i in range(len(population_intermediate)):
            population_intermediate[i].number = i

        # generate the phenotypic characterization of the individuals in the intermediate population
        start = time.time()
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate], dtype=object)
        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = np.array(decision_matrix)
        end = time.time()
        time_needed = end - start
        phenotypic_time_1.append(time_needed)

        # Create a duplicate of the intermediate population for the prediction
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_NB = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_LR = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_RF = [toolbox.clone(ind) for ind in population_intermediate]

        # Create a duplicate of the intermediate population for the estimation using simplified simulation
        population_intermediate_predicted_halfshop = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_singlerep = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_singlerep_short = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness using ML techniques
        start = time.time()
        prediction_KNN = predict(KNN_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        KNN_time_predict.append(time_needed)

        start = time.time()
        prediction_MLP = predict(MLP_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        MLP_time_predict.append(time_needed)

        start = time.time()
        prediction_DT = predict(DT_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        DT_time_predict.append(time_needed)

        start = time.time()
        prediction_SVM = predict(SVM_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        SVM_time_predict.append(time_needed)

        start = time.time()
        prediction_NB = predict(NB_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        NB_time_predict.append(time_needed)

        start = time.time()
        prediction_LR = predict(LR_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        LR_time_predict.append(time_needed)

        start = time.time()
        prediction_RF = predict(RF_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        RF_time_predict.append(time_needed)

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_NB]
        for ind, fit in zip(invalid_ind, prediction_NB):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_LR]
        for ind, fit in zip(invalid_ind, prediction_LR):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_RF]
        for ind, fit in zip(invalid_ind, prediction_RF):
            ind.fitness.values = fit

        # Estimation of the individuals fitness using simplified simulation
        # Half shop
        start = time.time()
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_halfshop], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_halfshop, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_halfshop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        end = time.time()
        time_needed = end - start
        HS_time_predict.append(time_needed)

        # Single Replication
        start = time.time()
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRep, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        end = time.time()
        time_needed = end - start
        SR_time_predict.append(time_needed)

        # Single Replication short
        start = time.time()
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep_short],
                               dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRepShort, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep_short]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        end = time.time()
        time_needed = end - start
        SR_short_time_predict.append(time_needed)


        # generate the ranking of the full populations (for each prdiction and actual)
        population_intermediate_full = toolbox.select(population_intermediate, pop_size_offspring)
        ranking_full = []
        for i in population_intermediate_full:
            ranking_full.append(i.number)
        #population_predicted_KNN_full = toolbox.select(population_intermediate_predicted_KNN, pop_size_offspring)
        #population_predicted_MLP_full = toolbox.select(population_intermediate_predicted_MLP, pop_size_offspring)
        #population_predicted_DT_full = toolbox.select(population_intermediate_predicted_DT, pop_size_offspring)
        #population_predicted_SVM_full = toolbox.select(population_intermediate_predicted_SVM, pop_size_offspring)
        #population_predicted_NB_full = toolbox.select(population_intermediate_predicted_NB, pop_size_offspring)
        #population_predicted_LR_full = toolbox.select(population_intermediate_predicted_LR, pop_size_offspring)
        #population_predicted_RF_full = toolbox.select(population_intermediate_predicted_RF, pop_size_offspring)


        # Select the next generation individuals based on the evaluation
        population_intermediate = toolbox.select(population_intermediate, pop_size)
        print(f'population size of intermediate population after selection: {len(population_intermediate)}')


        # Select the next generation individuals based on the estimation (just for evaluation purposes, no real selection is made)
        population_predicted_KNN = toolbox.select(population_intermediate_predicted_KNN, pop_size)
        population_predicted_MLP = toolbox.select(population_intermediate_predicted_MLP, pop_size)
        population_predicted_DT = toolbox.select(population_intermediate_predicted_DT, pop_size)
        population_predicted_SVM = toolbox.select(population_intermediate_predicted_SVM, pop_size)
        population_predicted_NB = toolbox.select(population_intermediate_predicted_NB, pop_size)
        population_predicted_LR = toolbox.select(population_intermediate_predicted_LR, pop_size)
        population_predicted_RF = toolbox.select(population_intermediate_predicted_RF, pop_size)

        # Select the next generation individuals based on the simplified simulations
        population_predicted_halfshop = toolbox.select(population_intermediate_predicted_halfshop, pop_size)
        population_predicted_singerep= toolbox.select(population_intermediate_predicted_singlerep, pop_size)
        population_predicted_singerep_short= toolbox.select(population_intermediate_predicted_singlerep_short, pop_size)

        # compare both selections and calculate the selection accuracy
        ranking = []
        ranking_predicted_KNN = []
        ranking_predicted_MLP = []
        ranking_predicted_DT = []
        ranking_predicted_SVM = []
        ranking_predicted_NB = []
        ranking_predicted_LR = []
        ranking_predicted_RF = []
        ranking_predicted_halfshop = []
        ranking_predicted_singlerep = []
        ranking_predicted_singlerep_short = []
        for i in population_intermediate:
            ranking.append(i.number)
        for i in population_predicted_KNN:
            ranking_predicted_KNN.append(i.number)
        for i in population_predicted_MLP:
            ranking_predicted_MLP.append(i.number)
        for i in population_predicted_DT:
            ranking_predicted_DT.append(i.number)
        for i in population_predicted_SVM:
            ranking_predicted_SVM.append(i.number)
        for i in population_predicted_NB:
            ranking_predicted_NB.append(i.number)
        for i in population_predicted_LR:
            ranking_predicted_LR.append(i.number)
        for i in population_predicted_RF:
            ranking_predicted_RF.append(i.number)
        for i in population_predicted_halfshop:
            ranking_predicted_halfshop.append(i.number)
        for i in population_predicted_singerep:
            ranking_predicted_singlerep.append(i.number)
        for i in population_predicted_singerep_short:
            ranking_predicted_singlerep_short.append(i.number)
        print(f'Ranking actual: {ranking}')
        print(f'Ranking KNN: {ranking_predicted_KNN}')
        print(f'Ranking MLP: {ranking_predicted_MLP}')
        print(f'Ranking DT: {ranking_predicted_DT}')
        print(f'Ranking SVM: {ranking_predicted_SVM}')
        print(f'Ranking NB: {ranking_predicted_NB}')
        print(f'Ranking LR: {ranking_predicted_LR}')
        print(f'Ranking RF: {ranking_predicted_RF}')
        print(f'Ranking halfshop: {ranking_predicted_halfshop}')
        print(f'Ranking single replication: {ranking_predicted_singlerep}')
        print(f'Ranking single replication short: {ranking_predicted_singlerep_short}')
        selection_accuracy_KNN, selection_accuracy_KNN_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_KNN,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_MLP, selection_accuracy_MLP_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_MLP,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_DT, selection_accuracy_DT_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_DT,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_SVM, selection_accuracy_SVM_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_SVM,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_NB, selection_accuracy_NB_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_NB,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_LR, selection_accuracy_LR_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_LR,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_RF, selection_accuracy_RF_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_RF,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_halfshop, selection_accuracy_halfshop_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_halfshop,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_singlerep, selection_accuracy_singlerep_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_singlerep,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_singlerep_short, selection_accuracy_singlerep_short_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_singlerep_short,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)

        # Evaluate the rank correlation
        rank_correlation_KNN = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_KNN,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_MLP = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_MLP,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_DT = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_DT,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_SVM = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_SVM,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_NB = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_NB,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_LR = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_LR,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_RF = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_RF,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_halfshop = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_halfshop,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_singlerep = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_singlerep,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_singlerep_short = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_singlerep_short,
                                                         pop_size_offspring=pop_size_offspring)



        selection_accuracy_list.append([selection_accuracy_KNN, selection_accuracy_MLP, selection_accuracy_DT,
                                        selection_accuracy_SVM, selection_accuracy_NB, selection_accuracy_LR,
                                        selection_accuracy_RF, selection_accuracy_halfshop, selection_accuracy_singlerep,
                                        selection_accuracy_singlerep_short])
        selection_accuracy_list_simple.append([selection_accuracy_KNN_simple, selection_accuracy_MLP_simple, selection_accuracy_DT_simple,
                                               selection_accuracy_SVM_simple, selection_accuracy_NB_simple, selection_accuracy_LR_simple,
                                               selection_accuracy_RF_simple, selection_accuracy_halfshop_simple, selection_accuracy_singlerep_simple,
                                               selection_accuracy_singlerep_short_simple])

        rank_correlation_list.append([rank_correlation_KNN, rank_correlation_MLP, rank_correlation_DT, rank_correlation_SVM,
                                      rank_correlation_NB, rank_correlation_LR, rank_correlation_RF, rank_correlation_halfshop,
                                      rank_correlation_singlerep, rank_correlation_singlerep_short])

        print('selection accuracy ')
        print(selection_accuracy_list)
        print('selection accuracy simple')
        print(selection_accuracy_list_simple)

        # remove the estimated fitness from the population to add
        #for ind in population_intermediate:
        #    del ind.fitness.values

        # add the pre-selected individuals to the population
        if gen < ngen:
            population += population_intermediate
        #print("\n")
        #print(f'New population:')
        #for i in population:
        #    print(str(i))

        print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Append the current generation statistics to the logbook
        #record = stats.compile(population) if stats else {}
        #logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        #if verbose:
        #    print(logbook.stream)

        # Update random seed
        rand_value = random.randint(1, 300)
        #end = time.time()
        #print(f'Execution time one loop: {end - start}')
    time_dict = {'phenotypic 1': phenotypic_time, 'phenotypic 2': phenotypic_time_1, 'KNN train': KNN_time_train, 'MLP train': MLP_time_train, 'DT train': DT_time_train, 'SVM train': SVM_time_train, 'NB train': NB_time_train,
                 'LR train': LR_time_train, 'RF train': RF_time_train, 'KNN predict': KNN_time_predict, 'MLP predict': MLP_time_predict, 'DT predict': DT_time_predict,
                 'SVM predict': SVM_time_predict, 'NB predict': NB_time_predict ,'LR predict': LR_time_predict, 'RF predict': RF_time_predict, 'SR': SR_time_predict,
                 'SR_short': SR_short_time_predict, 'HS': HS_time_predict, 'Full Evaluation': full_evaluation}
    return population, logbook, selection_accuracy_list, selection_accuracy_list_simple, rank_correlation_list, time_dict

# GPHH for testing the selection accuracy (using the latest 1000 fully evaluated individuals for training of surrogates
# optimized surrogate evaluation -> only generate the phenotypic characterization for not yet characterized individuals)
def GPHH_experiment1_new(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref,n , n_offspring, stats=None,
             halloffame=None, verbose=__debug__):


    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)
    '''
    # Evaluate the individuals with an invalid fitness using full evaluation
    invalid_ind = np.array([[ind, rand_value] for ind in population if not ind.fitness.valid], dtype=object)
    invalid_ind_1 = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind_1, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
'''

    # initialize the lists for measuring the time
    KNN_time_train = []
    MLP_time_train = []
    DT_time_train = []
    SVM_time_train = []
    NB_time_train = []
    LR_time_train = []
    RF_time_train = []

    KNN_time_predict = []
    MLP_time_predict = []
    DT_time_predict = []
    SVM_time_predict = []
    NB_time_predict = []
    LR_time_predict = []
    RF_time_predict = []

    SR_time_predict = []
    SR_short_time_predict = []
    HS_time_predict = []

    phenotypic_time = []
    phenotypic_time_1 = []

    full_evaluation = []

    # Update random seed
    rand_value = random.randint(1,300)

    # Initialize selection accuracy
    selection_accuracy_list = []
    selection_accuracy_list_simple = []
    rank_correlation_list = []

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population)/2)
    pop_size_offspring = int(len(population))
    #print(f'Initial population:')
    #for i in population:
    #    print(str(i))
    # Begin the generational process
    for gen in range(1, ngen + 1):
        print(f'Generation: {gen}')
        population = remove_duplicates(population=population)
        #start = time.time()
        print(f'population size: {len(population)}')
        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # add the full evaluated individuals to the training set
        pop_train += invalid_ind
        if len(pop_train) > 1000:
            pop_train = pop_train[-1000:]
        print(f'population size of the training population: {len(pop_train)}')
        #print("\n")
        #print(f'Training population:')
        #for i in pop_train:
        #    print(str(i))
        #print("\n")
        # generate the phenotypic characterization of the training individuals
        for i in pop_train:
            print(i.phenotypic)
        start = time.time()
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in pop_train if ind.phenotypic==None], dtype=object)
        invalid_ind_1 = [ind for ind in pop_train if ind.phenotypic==None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in pop_train]
        fitness_matrix = [ind.fitness.values for ind in pop_train]
        end = time.time()
        time_needed = end - start
        phenotypic_time.append(time_needed)
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')

        # Update the surrogate models (train the predictors)
        start = time.time()
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end-start
        KNN_time_train.append(time_needed)

        start = time.time()
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        MLP_time_train.append(time_needed)

        start = time.time()
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        DT_time_train.append(time_needed)

        start = time.time()
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        SVM_time_train.append(time_needed)

        start = time.time()
        NB_model = NB_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        NB_time_train.append(time_needed)

        start = time.time()
        LR_model = LR_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        LR_time_train.append(time_needed)

        start = time.time()
        RF_model = RF_train(X=decision_matrix, y=fitness_matrix)
        end = time.time()
        time_needed = end - start
        RF_time_train.append(time_needed)

        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        #print("\n")
        #print(f'Selected population:')
        #for i in population:
        #    print(str(i))
        #print("\n")
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = len(population)*n
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]

        #for i in range(n+1):
        #    offspring = varAnd(population, toolbox, cxpb, mutpb)
        #    population_intermediate += offspring
        #    #print("\n")

        #print("\n")
        #print(f'Intermediate population:')
        #for i in population_intermediate:
        #    print(str(i))
        #print("\n")

        # remove duplicates
        #population_intermediate = remove_duplicates(population=population_intermediate)
        # remove already evaluated individuals (duplicates to te original population after selection)
        #population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]


        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:(len(population)*n)]

        print(f'population size of intermediate population: {len(population_intermediate)}')

        # Evaluate the individuals with an invalid fitness using full evaluation (only for comparison purpose to calculate the accuracy)
        start = time.time()
        invalid_ind = [ind for ind in population_intermediate if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        end = time.time()
        time_needed = end - start
        full_evaluation.append(time_needed)

        print(f'evaluated individuals of intermediate population: {len(invalid_ind)}')
        # add the full evaluated individual to the training population
        pop_train += invalid_ind

        # Assign a number to each individual in the intermediate population
        for i in range(len(population_intermediate)):
            population_intermediate[i].number = i

        # generate the phenotypic characterization of the individuals in the intermediate population
        for i in population_intermediate:
            print(i.phenotypic)
        start = time.time()
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if ind.phenotypic==None], dtype=object)
        invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic==None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in population_intermediate]
        end = time.time()
        time_needed = end - start
        phenotypic_time_1.append(time_needed)
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')

        # Create a duplicate of the intermediate population for the prediction
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_NB = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_LR = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_RF = [toolbox.clone(ind) for ind in population_intermediate]

        # Create a duplicate of the intermediate population for the estimation using simplified simulation
        population_intermediate_predicted_halfshop = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_singlerep = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_singlerep_short = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness using ML techniques
        start = time.time()
        prediction_KNN = predict(KNN_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        KNN_time_predict.append(time_needed)

        start = time.time()
        prediction_MLP = predict(MLP_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        MLP_time_predict.append(time_needed)

        start = time.time()
        prediction_DT = predict(DT_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        DT_time_predict.append(time_needed)

        start = time.time()
        prediction_SVM = predict(SVM_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        SVM_time_predict.append(time_needed)

        start = time.time()
        prediction_NB = predict(NB_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        NB_time_predict.append(time_needed)

        start = time.time()
        prediction_LR = predict(LR_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        LR_time_predict.append(time_needed)

        start = time.time()
        prediction_RF = predict(RF_model, decision_matrix)
        end = time.time()
        time_needed = end - start
        RF_time_predict.append(time_needed)

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_NB]
        for ind, fit in zip(invalid_ind, prediction_NB):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_LR]
        for ind, fit in zip(invalid_ind, prediction_LR):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_RF]
        for ind, fit in zip(invalid_ind, prediction_RF):
            ind.fitness.values = fit

        # Estimation of the individuals fitness using simplified simulation
        # Half shop
        start = time.time()
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_halfshop], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_halfshop, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_halfshop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        end = time.time()
        time_needed = end - start
        HS_time_predict.append(time_needed)

        # Single Replication
        start = time.time()
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRep, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        end = time.time()
        time_needed = end - start
        SR_time_predict.append(time_needed)

        # Single Replication short
        start = time.time()
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep_short],
                               dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRepShort, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep_short]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        end = time.time()
        time_needed = end - start
        SR_short_time_predict.append(time_needed)


        # generate the ranking of the full populations (for each prdiction and actual)
        population_intermediate_full = toolbox.select(population_intermediate, pop_size_offspring)
        ranking_full = []
        for i in population_intermediate_full:
            ranking_full.append(i.number)
        #population_predicted_KNN_full = toolbox.select(population_intermediate_predicted_KNN, pop_size_offspring)
        #population_predicted_MLP_full = toolbox.select(population_intermediate_predicted_MLP, pop_size_offspring)
        #population_predicted_DT_full = toolbox.select(population_intermediate_predicted_DT, pop_size_offspring)
        #population_predicted_SVM_full = toolbox.select(population_intermediate_predicted_SVM, pop_size_offspring)
        #population_predicted_NB_full = toolbox.select(population_intermediate_predicted_NB, pop_size_offspring)
        #population_predicted_LR_full = toolbox.select(population_intermediate_predicted_LR, pop_size_offspring)
        #population_predicted_RF_full = toolbox.select(population_intermediate_predicted_RF, pop_size_offspring)


        # Select the next generation individuals based on the evaluation
        population_intermediate = toolbox.select(population_intermediate, pop_size)
        print(f'population size of intermediate population after selection: {len(population_intermediate)}')


        # Select the next generation individuals based on the estimation (just for evaluation purposes, no real selection is made)
        population_predicted_KNN = toolbox.select(population_intermediate_predicted_KNN, pop_size)
        population_predicted_MLP = toolbox.select(population_intermediate_predicted_MLP, pop_size)
        population_predicted_DT = toolbox.select(population_intermediate_predicted_DT, pop_size)
        population_predicted_SVM = toolbox.select(population_intermediate_predicted_SVM, pop_size)
        population_predicted_NB = toolbox.select(population_intermediate_predicted_NB, pop_size)
        population_predicted_LR = toolbox.select(population_intermediate_predicted_LR, pop_size)
        population_predicted_RF = toolbox.select(population_intermediate_predicted_RF, pop_size)

        # Select the next generation individuals based on the simplified simulations
        population_predicted_halfshop = toolbox.select(population_intermediate_predicted_halfshop, pop_size)
        population_predicted_singerep= toolbox.select(population_intermediate_predicted_singlerep, pop_size)
        population_predicted_singerep_short= toolbox.select(population_intermediate_predicted_singlerep_short, pop_size)

        # compare both selections and calculate the selection accuracy
        ranking = []
        ranking_predicted_KNN = []
        ranking_predicted_MLP = []
        ranking_predicted_DT = []
        ranking_predicted_SVM = []
        ranking_predicted_NB = []
        ranking_predicted_LR = []
        ranking_predicted_RF = []
        ranking_predicted_halfshop = []
        ranking_predicted_singlerep = []
        ranking_predicted_singlerep_short = []
        for i in population_intermediate:
            ranking.append(i.number)
        for i in population_predicted_KNN:
            ranking_predicted_KNN.append(i.number)
        for i in population_predicted_MLP:
            ranking_predicted_MLP.append(i.number)
        for i in population_predicted_DT:
            ranking_predicted_DT.append(i.number)
        for i in population_predicted_SVM:
            ranking_predicted_SVM.append(i.number)
        for i in population_predicted_NB:
            ranking_predicted_NB.append(i.number)
        for i in population_predicted_LR:
            ranking_predicted_LR.append(i.number)
        for i in population_predicted_RF:
            ranking_predicted_RF.append(i.number)
        for i in population_predicted_halfshop:
            ranking_predicted_halfshop.append(i.number)
        for i in population_predicted_singerep:
            ranking_predicted_singlerep.append(i.number)
        for i in population_predicted_singerep_short:
            ranking_predicted_singlerep_short.append(i.number)
        print(f'Ranking actual: {ranking}')
        print(f'Ranking KNN: {ranking_predicted_KNN}')
        print(f'Ranking MLP: {ranking_predicted_MLP}')
        print(f'Ranking DT: {ranking_predicted_DT}')
        print(f'Ranking SVM: {ranking_predicted_SVM}')
        print(f'Ranking NB: {ranking_predicted_NB}')
        print(f'Ranking LR: {ranking_predicted_LR}')
        print(f'Ranking RF: {ranking_predicted_RF}')
        print(f'Ranking halfshop: {ranking_predicted_halfshop}')
        print(f'Ranking single replication: {ranking_predicted_singlerep}')
        print(f'Ranking single replication short: {ranking_predicted_singlerep_short}')
        selection_accuracy_KNN, selection_accuracy_KNN_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_KNN,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_MLP, selection_accuracy_MLP_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_MLP,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_DT, selection_accuracy_DT_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_DT,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_SVM, selection_accuracy_SVM_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_SVM,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_NB, selection_accuracy_NB_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_NB,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_LR, selection_accuracy_LR_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_LR,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_RF, selection_accuracy_RF_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_RF,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_halfshop, selection_accuracy_halfshop_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_halfshop,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_singlerep, selection_accuracy_singlerep_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_singlerep,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_singlerep_short, selection_accuracy_singlerep_short_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_singlerep_short,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)

        # Evaluate the rank correlation
        rank_correlation_KNN = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_KNN,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_MLP = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_MLP,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_DT = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_DT,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_SVM = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_SVM,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_NB = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_NB,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_LR = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_LR,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_RF = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_RF,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_halfshop = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_halfshop,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_singlerep = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_singlerep,
                                                         pop_size_offspring=pop_size_offspring)
        rank_correlation_singlerep_short = evaluate_rank_correlation(ranking_actual=ranking_full,
                                                         ranking_predicted=ranking_predicted_singlerep_short,
                                                         pop_size_offspring=pop_size_offspring)



        selection_accuracy_list.append([selection_accuracy_KNN, selection_accuracy_MLP, selection_accuracy_DT,
                                        selection_accuracy_SVM, selection_accuracy_NB, selection_accuracy_LR,
                                        selection_accuracy_RF, selection_accuracy_halfshop, selection_accuracy_singlerep,
                                        selection_accuracy_singlerep_short])
        selection_accuracy_list_simple.append([selection_accuracy_KNN_simple, selection_accuracy_MLP_simple, selection_accuracy_DT_simple,
                                               selection_accuracy_SVM_simple, selection_accuracy_NB_simple, selection_accuracy_LR_simple,
                                               selection_accuracy_RF_simple, selection_accuracy_halfshop_simple, selection_accuracy_singlerep_simple,
                                               selection_accuracy_singlerep_short_simple])

        rank_correlation_list.append([rank_correlation_KNN, rank_correlation_MLP, rank_correlation_DT, rank_correlation_SVM,
                                      rank_correlation_NB, rank_correlation_LR, rank_correlation_RF, rank_correlation_halfshop,
                                      rank_correlation_singlerep, rank_correlation_singlerep_short])

        print('selection accuracy ')
        print(selection_accuracy_list)
        print('selection accuracy simple')
        print(selection_accuracy_list_simple)

        # remove the estimated fitness from the population to add
        #for ind in population_intermediate:
        #    del ind.fitness.values

        # add the pre-selected individuals to the population
        if gen < ngen:
            population += population_intermediate
        #print("\n")
        #print(f'New population:')
        #for i in population:
        #    print(str(i))

        print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Append the current generation statistics to the logbook
        #record = stats.compile(population) if stats else {}
        #logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        #if verbose:
        #    print(logbook.stream)

        # Update random seed
        rand_value = random.randint(1, 300)
        #end = time.time()
        #print(f'Execution time one loop: {end - start}')
    time_dict = {'phenotypic 1': phenotypic_time, 'phenotypic 2': phenotypic_time_1, 'KNN train': KNN_time_train, 'MLP train': MLP_time_train, 'DT train': DT_time_train, 'SVM train': SVM_time_train, 'NB train': NB_time_train,
                 'LR train': LR_time_train, 'RF train': RF_time_train, 'KNN predict': KNN_time_predict, 'MLP predict': MLP_time_predict, 'DT predict': DT_time_predict,
                 'SVM predict': SVM_time_predict, 'NB predict': NB_time_predict ,'LR predict': LR_time_predict, 'RF predict': RF_time_predict, 'SR': SR_time_predict,
                 'SR_short': SR_short_time_predict, 'HS': HS_time_predict, 'Full Evaluation': full_evaluation}
    return population, logbook, selection_accuracy_list, selection_accuracy_list_simple, rank_correlation_list, time_dict


# GPHH with surrogates (using the latest 1000 fully evaluated individuals for training of surrogates)
# optimized surrogate evaluation -> only generate the phenotypic characterization for not yet characterized individuals)
def GPHH_experiment2_SR(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref,n , n_offspring, stats=None,
             halloffame=None, verbose=__debug__):

    # Initialize count for total time
    start_total_time = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)

    # initialize the lists for measuring the time
    time_per_generation = []

    # Update random seed
    rand_value = random.randint(1,300)

    # Initialize lists for populations
    population_dict = {}
    fitness_dict = {}

    # Initialize the list for tracking the number of fitness evaluations
    fitness_evaluations = {}

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population)/2)
    pop_size_offspring = int(len(population))

    # Begin the generational process
    for gen in range(1, ngen + 1):
        start = time.time()
        print(f'Generation: {gen}')
        population = remove_duplicates(population=population)
        print(f'population size: {len(population)}')

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the number of fitness evaluations to the dictionary
        full_evaluations = len(invalid_ind)*5
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # save the full evaluated individuals to the population dict
        population_dict[f'Individuals Generation {gen}'] = [str(ind) for ind in population]

        fitness_list = [str(ind.fitness) for ind in population]
        tuple_list = [tuple(float(s) for s in x.strip("()").split(",")) for x in fitness_list]
        f1_list = [a_tuple[0] for a_tuple in tuple_list]
        f2_list = [a_tuple[1] for a_tuple in tuple_list]
        fitness_dict[f'Fitness f1 Generation {gen}'] = f1_list
        fitness_dict[f'Fitness f2 Generation {gen}'] = f2_list
        '''
        # add the full evaluated individuals to the training set
        pop_train += invalid_ind
        if len(pop_train) > 1000:
            pop_train = pop_train[-1000:]
        print(f'population size of the training population: {len(pop_train)}')

        # generate the phenotypic characterization of the training individuals
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in pop_train if ind.phenotypic==None], dtype=object)
        invalid_ind_1 = [ind for ind in pop_train if ind.phenotypic==None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in pop_train]
        fitness_matrix = [ind.fitness.values for ind in pop_train]
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')
        '''
        # Update the surrogate models (train the predictors)
        '''
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)
        NB_model = NB_train(X=decision_matrix, y=fitness_matrix)
        LR_model = LR_train(X=decision_matrix, y=fitness_matrix)
        RF_model = RF_train(X=decision_matrix, y=fitness_matrix)
        '''
        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = len(population)*n
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]

        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:(len(population)*n)]
        print(f'population size of intermediate population: {len(population_intermediate)}')

        '''
        # generate the phenotypic characterization of the individuals in the intermediate population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if ind.phenotypic==None], dtype=object)
        invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic==None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in population_intermediate]
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')
        '''
        # Create a duplicate of the intermediate population for the prediction
        '''
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_NB = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_LR = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_RF = [toolbox.clone(ind) for ind in population_intermediate]
        '''
        # Create a duplicate of the intermediate population for the estimation using simplified simulation

        #population_intermediate_predicted_halfshop = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_singlerep = [toolbox.clone(ind) for ind in population_intermediate]
        #population_intermediate_predicted_singlerep_short = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness using ML techniques
        '''
        prediction_KNN = predict(KNN_model, decision_matrix)
        prediction_MLP = predict(MLP_model, decision_matrix)
        prediction_DT = predict(DT_model, decision_matrix)
        prediction_SVM = predict(SVM_model, decision_matrix)
        prediction_NB = predict(NB_model, decision_matrix)
        prediction_LR = predict(LR_model, decision_matrix)
        prediction_RF = predict(RF_model, decision_matrix)
        '''

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        '''
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_NB]
        for ind, fit in zip(invalid_ind, prediction_NB):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_LR]
        for ind, fit in zip(invalid_ind, prediction_LR):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_RF]
        for ind, fit in zip(invalid_ind, prediction_RF):
            ind.fitness.values = fit
        '''
        # Estimation of the individuals fitness using simplified simulation
        '''
        # Half shop
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_halfshop], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_halfshop, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_halfshop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        '''
        # Single Replication
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRep, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the number of fitness evaluations to the dictionary
        total_evaluations = full_evaluations+len(invalid_ind)
        fitness_evaluations[f'Single evaluated individuals generation {gen}'] = total_evaluations
        print(f'Single evaluated individuals: {len(invalid_ind)}')

        '''
        # Single Replication short
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep_short],
                               dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRepShort, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep_short]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        '''

        # Select the next generation individuals based on the prediction
        population_intermediate = toolbox.select(population_intermediate_predicted_singlerep, pop_size)
        print(f'population size of intermediate population after selection: {len(population_intermediate)}')

        # remove the estimated fitness from the population to add
        for ind in population_intermediate:
            del ind.fitness.values

        # add the pre-selected individuals to the population
        if gen < ngen:
            population += population_intermediate

        print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Update random seed
        rand_value = random.randint(1, 300)
        end = time.time()
        time_current_generation = end-start
        time_per_generation.append(time_current_generation)
        print(f'Execution time one loop: {time_current_generation}')

    end_total_time = time.time()
    total_time = end_total_time-start_total_time
    time_dict = {'time per generation': time_per_generation, 'total time': total_time}
    return population, logbook, time_dict, population_dict, fitness_dict, fitness_evaluations

# GPHH with surrogates (using the latest 1000 fully evaluated individuals for training of surrogates)
# optimized surrogate evaluation -> only generate the phenotypic characterization for not yet characterized individuals)
def GPHH_experiment2_RF(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref,n , n_offspring, stats=None,
             halloffame=None, verbose=__debug__):

    # Initialize count for total time
    start_total_time = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)

    # initialize the lists for measuring the time
    time_per_generation = []

    # Update random seed
    rand_value = random.randint(1,300)

    # Initialize lists for populations
    population_dict = {}
    fitness_dict = {}

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population)/2)
    pop_size_offspring = int(len(population))

    # Begin the generational process
    for gen in range(1, ngen + 1):
        #n_adaptive = (ngen-gen+1)/ngen * n +1
        start = time.time()
        print(f'Generation: {gen}')
        #print(f'n adaptive: {n_adaptive}')
        population = remove_duplicates(population=population)
        print(f'population size: {len(population)}')

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # save the full evaluated individuals to the population dict
        population_dict[f'Individuals Generation {gen}'] = [str(ind) for ind in population]

        fitness_list = [str(ind.fitness) for ind in population]
        tuple_list = [tuple(float(s) for s in x.strip("()").split(",")) for x in fitness_list]
        f1_list = [a_tuple[0] for a_tuple in tuple_list]
        f2_list = [a_tuple[1] for a_tuple in tuple_list]
        fitness_dict[f'Fitness f1 Generation {gen}'] = f1_list
        fitness_dict[f'Fitness f2 Generation {gen}'] = f2_list

        # add the full evaluated individuals to the training set
        pop_train += invalid_ind
        if len(pop_train) > 1000:
            pop_train = pop_train[-1000:]
        print(f'population size of the training population: {len(pop_train)}')

        # generate the phenotypic characterization of the training individuals
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in pop_train if ind.phenotypic==None], dtype=object)
        invalid_ind_1 = [ind for ind in pop_train if ind.phenotypic==None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in pop_train]
        fitness_matrix = [ind.fitness.values for ind in pop_train]
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')

        # Update the surrogate models (train the predictors)
        '''
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)
        NB_model = NB_train(X=decision_matrix, y=fitness_matrix)
        LR_model = LR_train(X=decision_matrix, y=fitness_matrix)
        '''
        RF_model = RF_train(X=decision_matrix, y=fitness_matrix)

        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = int(len(population)*n)
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]

        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:population_size_intermediate]
        print(f'population size of intermediate population: {len(population_intermediate)}')


        # generate the phenotypic characterization of the individuals in the intermediate population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if ind.phenotypic==None], dtype=object)
        invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic==None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in population_intermediate]
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')

        # Create a duplicate of the intermediate population for the prediction
        '''
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_NB = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_LR = [toolbox.clone(ind) for ind in population_intermediate]
        '''
        population_intermediate_predicted_RF = [toolbox.clone(ind) for ind in population_intermediate]

        # Create a duplicate of the intermediate population for the estimation using simplified simulation

        #population_intermediate_predicted_halfshop = [toolbox.clone(ind) for ind in population_intermediate]
        #population_intermediate_predicted_singlerep = [toolbox.clone(ind) for ind in population_intermediate]
        #population_intermediate_predicted_singlerep_short = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness using ML techniques
        '''
        prediction_KNN = predict(KNN_model, decision_matrix)
        prediction_MLP = predict(MLP_model, decision_matrix)
        prediction_DT = predict(DT_model, decision_matrix)
        prediction_SVM = predict(SVM_model, decision_matrix)
        prediction_NB = predict(NB_model, decision_matrix)
        prediction_LR = predict(LR_model, decision_matrix)
        '''
        prediction_RF = predict(RF_model, decision_matrix)


        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        '''
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_NB]
        for ind, fit in zip(invalid_ind, prediction_NB):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_LR]
        for ind, fit in zip(invalid_ind, prediction_LR):
            ind.fitness.values = fit
        '''
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_RF]
        for ind, fit in zip(invalid_ind, prediction_RF):
            ind.fitness.values = fit

        # Estimation of the individuals fitness using simplified simulation
        '''
        # Half shop
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_halfshop], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_halfshop, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_halfshop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Single Replication
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRep, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Single Replication short
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep_short],
                               dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRepShort, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep_short]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        '''

        # Select the next generation individuals based on the prediction
        population_intermediate = toolbox.select(population_intermediate_predicted_RF, pop_size)
        print(f'population size of intermediate population after selection: {len(population_intermediate)}')

        # remove the estimated fitness from the population to add
        for ind in population_intermediate:
            del ind.fitness.values

        # add the pre-selected individuals to the population
        if gen < ngen:
            population += population_intermediate

        print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Update random seed
        rand_value = random.randint(1, 300)
        end = time.time()
        time_current_generation = end-start
        time_per_generation.append(time_current_generation)
        print(f'Execution time one loop: {time_current_generation}')

    end_total_time = time.time()
    total_time = end_total_time-start_total_time
    time_dict = {'time per generation': time_per_generation, 'total time': total_time}
    return population, logbook, time_dict, population_dict, fitness_dict


# GPHH with surrogates (using the latest 1000 fully evaluated individuals for training of surrogates)
# optimized surrogate evaluation -> only generate the phenotypic characterization for not yet characterized individuals)
def GPHH_experiment2_RF_duplicate(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref, n, n_offspring,
                        stats=None,
                        halloffame=None, verbose=__debug__):
    # Initialize count for total time
    start_total_time = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1, 300)

    # initialize the lists for measuring the time
    time_per_generation = []

    # Update random seed
    rand_value = random.randint(1, 300)

    # Initialize lists for populations
    population_dict = {}
    fitness_dict = {}

    # Initialize the list for tracking the number of fitness evaluations
    fitness_evaluations = {}

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population) / 2)
    pop_size_offspring = int(len(population))

    # generate phenotypic characterization of the initial population
    invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population if
                            ind.phenotypic == None], dtype=object)
    invalid_ind_1 = [ind for ind in population if ind.phenotypic == None]
    phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
    for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
        ind.phenotypic = phenotypic
    print(f'original population size: {len(population)}')
    # remove duplicates from the initial population based on the phenotypic characterization
    population = remove_duplicates_phenotypic(population=population)
    print(f'population size after removing duplicates (phenotypic): {len(population)}')

    while len(population) < pop_size_offspring:
        # produce new individuals
        new_pop = toolbox.population(n=200)
        # add new individuals to the initial population
        population += new_pop
        print(f'population size after adding new individuals: {len(population)}')
        # generate phenotypic characterization of the initial population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population if
                                ind.phenotypic == None], dtype=object)
        invalid_ind_1 = [ind for ind in population if ind.phenotypic == None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        population = remove_duplicates_phenotypic(population=population)
        print(f'population size after removing duplicates (phenotypic): {len(population)}')

    population = population[:pop_size_offspring]

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # n_adaptive = (ngen-gen+1)/ngen * n +1
        start = time.time()
        print(f'Generation: {gen}')
        # print(f'n adaptive: {n_adaptive}')
        population = remove_duplicates_phenotypic(population=population)
        print(f'population size: {len(population)}')

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the number of fitness evaluations to the dictionary
        fitness_evaluations[f'Full evaluated individuals generation {gen}'] = len(invalid_ind)
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # save the full evaluated individuals to the population dict
        population_dict[f'Individuals Generation {gen}'] = [str(ind) for ind in population]

        fitness_list = [str(ind.fitness) for ind in population]
        tuple_list = [tuple(float(s) for s in x.strip("()").split(",")) for x in fitness_list]
        f1_list = [a_tuple[0] for a_tuple in tuple_list]
        f2_list = [a_tuple[1] for a_tuple in tuple_list]
        fitness_dict[f'Fitness f1 Generation {gen}'] = f1_list
        fitness_dict[f'Fitness f2 Generation {gen}'] = f2_list

        # add the full evaluated individuals to the training set
        pop_train += invalid_ind
        if len(pop_train) > 1000:
            pop_train = pop_train[-1000:]
        print(f'population size of the training population: {len(pop_train)}')

        # generate the phenotypic characterization of the training individuals
        invalid_ind = np.array(
            [[ind, decision_situations, ranking_vector_ref] for ind in pop_train if ind.phenotypic == None],
            dtype=object)
        invalid_ind_1 = [ind for ind in pop_train if ind.phenotypic == None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in pop_train]
        fitness_matrix = [ind.fitness.values for ind in pop_train]
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')

        # Update the surrogate models (train the predictors)
        '''
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)
        NB_model = NB_train(X=decision_matrix, y=fitness_matrix)
        LR_model = LR_train(X=decision_matrix, y=fitness_matrix)
        '''
        RF_model = RF_train(X=decision_matrix, y=fitness_matrix)

        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = int(len(population) * n)
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            #population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]
            # generate the phenotypic characterization of the individuals in the intermediate population
            invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if
                                    ind.phenotypic == None], dtype=object)
            invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic == None]
            phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
            for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
                ind.phenotypic = phenotypic
            print(f'Population size before removing the duplicates (phenotypic): {len(population_intermediate)}')
            population_intermediate = remove_duplicates_phenotypic(population=population_intermediate)
            print(f'Population size after removing the duplicates (phenotypic): {len(population_intermediate)}')

        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:population_size_intermediate]
        print(f'population size of intermediate population: {len(population_intermediate)}')

        # generate the phenotypic characterization of the individuals in the intermediate population
        #invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if
        #                        ind.phenotypic == None], dtype=object)
        #invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic == None]
        #phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        #for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
        #    ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in population_intermediate]
        #print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')2

        # Create a duplicate of the intermediate population for the prediction
        '''
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_NB = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_LR = [toolbox.clone(ind) for ind in population_intermediate]
        '''
        population_intermediate_predicted_RF = [toolbox.clone(ind) for ind in population_intermediate]

        # Create a duplicate of the intermediate population for the estimation using simplified simulation

        # population_intermediate_predicted_halfshop = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_singlerep = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_singlerep_short = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness using ML techniques
        '''
        prediction_KNN = predict(KNN_model, decision_matrix)
        prediction_MLP = predict(MLP_model, decision_matrix)
        prediction_DT = predict(DT_model, decision_matrix)
        prediction_SVM = predict(SVM_model, decision_matrix)
        prediction_NB = predict(NB_model, decision_matrix)
        prediction_LR = predict(LR_model, decision_matrix)
        '''
        prediction_RF = predict(RF_model, decision_matrix)

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        '''
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_NB]
        for ind, fit in zip(invalid_ind, prediction_NB):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_LR]
        for ind, fit in zip(invalid_ind, prediction_LR):
            ind.fitness.values = fit
        '''
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_RF]
        for ind, fit in zip(invalid_ind, prediction_RF):
            ind.fitness.values = fit

        # Estimation of the individuals fitness using simplified simulation
        '''
        # Half shop
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_halfshop], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_halfshop, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_halfshop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Single Replication
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRep, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Single Replication short
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep_short],
                               dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRepShort, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep_short]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        '''

        # Select the next generation individuals based on the prediction
        population_intermediate = toolbox.select(population_intermediate_predicted_RF, pop_size)
        print(f'population size of intermediate population after selection: {len(population_intermediate)}')

        # remove the estimated fitness from the population to add
        for ind in population_intermediate:
            del ind.fitness.values

        # add the pre-selected individuals to the population
        if gen < ngen:
            population += population_intermediate

        print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Update random seed
        rand_value = random.randint(1, 300)
        end = time.time()
        time_current_generation = end - start
        time_per_generation.append(time_current_generation)
        print(f'Execution time one loop: {time_current_generation}')

    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    time_dict = {'time per generation': time_per_generation, 'total time': total_time}
    return population, logbook, time_dict, population_dict, fitness_dict, fitness_evaluations

# GPHH with surrogates (using the latest 1000 fully evaluated individuals for training of surrogates)
# optimized surrogate evaluation -> only generate the phenotypic characterization for not yet characterized individuals)
def GPHH_experiment2_DT_duplicate(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref, n, n_offspring,
                        stats=None,
                        halloffame=None, verbose=__debug__):
    # Initialize count for total time
    start_total_time = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1, 300)

    # initialize the lists for measuring the time
    time_per_generation = []

    # Update random seed
    rand_value = random.randint(1, 300)

    # Initialize lists for populations
    population_dict = {}
    fitness_dict = {}

    # Initialize the list for tracking the number of fitness evaluations
    fitness_evaluations = {}

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population) / 2)
    pop_size_offspring = int(len(population))

    # generate phenotypic characterization of the initial population
    invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population if
                            ind.phenotypic == None], dtype=object)
    invalid_ind_1 = [ind for ind in population if ind.phenotypic == None]
    phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
    for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
        ind.phenotypic = phenotypic
    print(f'original population size: {len(population)}')
    # remove duplicates from the initial population based on the phenotypic characterization
    population = remove_duplicates_phenotypic(population=population)
    print(f'population size after removing duplicates (phenotypic): {len(population)}')

    while len(population) < pop_size_offspring:
        # produce new individuals
        new_pop = toolbox.population(n=200)
        # add new individuals to the initial population
        population += new_pop
        print(f'population size after adding new individuals: {len(population)}')
        # generate phenotypic characterization of the initial population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population if
                                ind.phenotypic == None], dtype=object)
        invalid_ind_1 = [ind for ind in population if ind.phenotypic == None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        population = remove_duplicates_phenotypic(population=population)
        print(f'population size after removing duplicates (phenotypic): {len(population)}')

    population = population[:pop_size_offspring]

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # n_adaptive = (ngen-gen+1)/ngen * n +1
        start = time.time()
        print(f'Generation: {gen}')
        # print(f'n adaptive: {n_adaptive}')
        population = remove_duplicates_phenotypic(population=population)
        print(f'population size: {len(population)}')

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the number of fitness evaluations to the dictionary
        fitness_evaluations[f'Full evaluated individuals generation {gen}'] = len(invalid_ind)
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # save the full evaluated individuals to the population dict
        population_dict[f'Individuals Generation {gen}'] = [str(ind) for ind in population]

        fitness_list = [str(ind.fitness) for ind in population]
        tuple_list = [tuple(float(s) for s in x.strip("()").split(",")) for x in fitness_list]
        f1_list = [a_tuple[0] for a_tuple in tuple_list]
        f2_list = [a_tuple[1] for a_tuple in tuple_list]
        fitness_dict[f'Fitness f1 Generation {gen}'] = f1_list
        fitness_dict[f'Fitness f2 Generation {gen}'] = f2_list

        # add the full evaluated individuals to the training set
        pop_train += invalid_ind
        if len(pop_train) > 1000:
            pop_train = pop_train[-1000:]
        print(f'population size of the training population: {len(pop_train)}')

        # generate the phenotypic characterization of the training individuals
        invalid_ind = np.array(
            [[ind, decision_situations, ranking_vector_ref] for ind in pop_train if ind.phenotypic == None],
            dtype=object)
        invalid_ind_1 = [ind for ind in pop_train if ind.phenotypic == None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in pop_train]
        fitness_matrix = [ind.fitness.values for ind in pop_train]
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')

        # Update the surrogate models (train the predictors)
        '''
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)
        NB_model = NB_train(X=decision_matrix, y=fitness_matrix)
        LR_model = LR_train(X=decision_matrix, y=fitness_matrix)
        
        RF_model = RF_train(X=decision_matrix, y=fitness_matrix)
        '''
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = int(len(population) * n)
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            #population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]
            # generate the phenotypic characterization of the individuals in the intermediate population
            invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if
                                    ind.phenotypic == None], dtype=object)
            invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic == None]
            phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
            for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
                ind.phenotypic = phenotypic
            print(f'Population size before removing the duplicates (phenotypic): {len(population_intermediate)}')
            population_intermediate = remove_duplicates_phenotypic(population=population_intermediate)
            print(f'Population size after removing the duplicates (phenotypic): {len(population_intermediate)}')

        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:population_size_intermediate]
        print(f'population size of intermediate population: {len(population_intermediate)}')

        # generate the phenotypic characterization of the individuals in the intermediate population
        #invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if
        #                        ind.phenotypic == None], dtype=object)
        #invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic == None]
        #phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        #for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
        #    ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in population_intermediate]
        #print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')2

        # Create a duplicate of the intermediate population for the prediction
        '''
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_RF = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_NB = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_LR = [toolbox.clone(ind) for ind in population_intermediate]
        '''

        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        # Create a duplicate of the intermediate population for the estimation using simplified simulation

        # population_intermediate_predicted_halfshop = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_singlerep = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_singlerep_short = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness using ML techniques
        '''
        prediction_KNN = predict(KNN_model, decision_matrix)
        prediction_MLP = predict(MLP_model, decision_matrix)
        prediction_RF = predict(RF_model, decision_matrix)
        prediction_SVM = predict(SVM_model, decision_matrix)
        prediction_NB = predict(NB_model, decision_matrix)
        prediction_LR = predict(LR_model, decision_matrix)
        '''

        prediction_DT = predict(DT_model, decision_matrix)

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        '''
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_RF]
        for ind, fit in zip(invalid_ind, prediction_RF):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_NB]
        for ind, fit in zip(invalid_ind, prediction_NB):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_LR]
        for ind, fit in zip(invalid_ind, prediction_LR):
            ind.fitness.values = fit
        '''


        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit

        # Estimation of the individuals fitness using simplified simulation
        '''
        # Half shop
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_halfshop], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_halfshop, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_halfshop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Single Replication
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRep, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Single Replication short
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep_short],
                               dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRepShort, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep_short]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        '''

        # Select the next generation individuals based on the prediction
        population_intermediate = toolbox.select(population_intermediate_predicted_DT, pop_size)
        print(f'population size of intermediate population after selection: {len(population_intermediate)}')

        # remove the estimated fitness from the population to add
        for ind in population_intermediate:
            del ind.fitness.values

        # add the pre-selected individuals to the population
        if gen < ngen:
            population += population_intermediate

        print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Update random seed
        rand_value = random.randint(1, 300)
        end = time.time()
        time_current_generation = end - start
        time_per_generation.append(time_current_generation)
        print(f'Execution time one loop: {time_current_generation}')

    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    time_dict = {'time per generation': time_per_generation, 'total time': total_time}
    return population, logbook, time_dict, population_dict, fitness_dict, fitness_evaluations


# GPHH with surrogates (using the latest 1000 fully evaluated individuals for training of surrogates)
# optimized surrogate evaluation -> only generate the phenotypic characterization for not yet characterized individuals)
def GPHH_experiment2_NB_duplicate(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref, n,
                                  n_offspring,
                                  stats=None,
                                  halloffame=None, verbose=__debug__):
    # Initialize count for total time
    start_total_time = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1, 300)

    # initialize the lists for measuring the time
    time_per_generation = []

    # Update random seed
    rand_value = random.randint(1, 300)

    # Initialize lists for populations
    population_dict = {}
    fitness_dict = {}

    # Initialize the list for tracking the number of fitness evaluations
    fitness_evaluations = {}

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population) / 2)
    pop_size_offspring = int(len(population))

    # generate phenotypic characterization of the initial population
    invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population if
                            ind.phenotypic == None], dtype=object)
    invalid_ind_1 = [ind for ind in population if ind.phenotypic == None]
    phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
    for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
        ind.phenotypic = phenotypic
    print(f'original population size: {len(population)}')
    # remove duplicates from the initial population based on the phenotypic characterization
    population = remove_duplicates_phenotypic(population=population)
    print(f'population size after removing duplicates (phenotypic): {len(population)}')

    while len(population) < pop_size_offspring:
        # produce new individuals
        new_pop = toolbox.population(n=200)
        # add new individuals to the initial population
        population += new_pop
        print(f'population size after adding new individuals: {len(population)}')
        # generate phenotypic characterization of the initial population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population if
                                ind.phenotypic == None], dtype=object)
        invalid_ind_1 = [ind for ind in population if ind.phenotypic == None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        population = remove_duplicates_phenotypic(population=population)
        print(f'population size after removing duplicates (phenotypic): {len(population)}')

    population = population[:pop_size_offspring]

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # n_adaptive = (ngen-gen+1)/ngen * n +1
        start = time.time()
        print(f'Generation: {gen}')
        # print(f'n adaptive: {n_adaptive}')
        population = remove_duplicates_phenotypic(population=population)
        print(f'population size: {len(population)}')

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the number of fitness evaluations to the dictionary
        fitness_evaluations[f'Full evaluated individuals generation {gen}'] = len(invalid_ind)
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # save the full evaluated individuals to the population dict
        population_dict[f'Individuals Generation {gen}'] = [str(ind) for ind in population]

        fitness_list = [str(ind.fitness) for ind in population]
        tuple_list = [tuple(float(s) for s in x.strip("()").split(",")) for x in fitness_list]
        f1_list = [a_tuple[0] for a_tuple in tuple_list]
        f2_list = [a_tuple[1] for a_tuple in tuple_list]
        fitness_dict[f'Fitness f1 Generation {gen}'] = f1_list
        fitness_dict[f'Fitness f2 Generation {gen}'] = f2_list

        # add the full evaluated individuals to the training set
        pop_train += invalid_ind
        if len(pop_train) > 1000:
            pop_train = pop_train[-1000:]
        print(f'population size of the training population: {len(pop_train)}')

        # generate the phenotypic characterization of the training individuals
        invalid_ind = np.array(
            [[ind, decision_situations, ranking_vector_ref] for ind in pop_train if ind.phenotypic == None],
            dtype=object)
        invalid_ind_1 = [ind for ind in pop_train if ind.phenotypic == None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in pop_train]
        fitness_matrix = [ind.fitness.values for ind in pop_train]
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')

        # Update the surrogate models (train the predictors)
        '''
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)

        LR_model = LR_train(X=decision_matrix, y=fitness_matrix)
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        RF_model = RF_train(X=decision_matrix, y=fitness_matrix)
        '''
        NB_model = NB_train(X=decision_matrix, y=fitness_matrix)
        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = int(len(population) * n)
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            # population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]
            # generate the phenotypic characterization of the individuals in the intermediate population
            invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if
                                    ind.phenotypic == None], dtype=object)
            invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic == None]
            phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
            for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
                ind.phenotypic = phenotypic
            print(f'Population size before removing the duplicates (phenotypic): {len(population_intermediate)}')
            population_intermediate = remove_duplicates_phenotypic(population=population_intermediate)
            print(f'Population size after removing the duplicates (phenotypic): {len(population_intermediate)}')

        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:population_size_intermediate]
        print(f'population size of intermediate population: {len(population_intermediate)}')

        # generate the phenotypic characterization of the individuals in the intermediate population
        # invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if
        #                        ind.phenotypic == None], dtype=object)
        # invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic == None]
        # phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        # for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
        #    ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in population_intermediate]
        # print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')2

        # Create a duplicate of the intermediate population for the prediction
        '''
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_RF = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_LR = [toolbox.clone(ind) for ind in population_intermediate]
        '''
        population_intermediate_predicted_NB = [toolbox.clone(ind) for ind in population_intermediate]

        # Create a duplicate of the intermediate population for the estimation using simplified simulation

        # population_intermediate_predicted_halfshop = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_singlerep = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_singlerep_short = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness using ML techniques
        '''
        prediction_KNN = predict(KNN_model, decision_matrix)
        prediction_MLP = predict(MLP_model, decision_matrix)
        prediction_RF = predict(RF_model, decision_matrix)
        prediction_SVM = predict(SVM_model, decision_matrix)
        prediction_DT = predict(DT_model, decision_matrix)
        prediction_LR = predict(LR_model, decision_matrix)
        '''
        prediction_NB = predict(NB_model, decision_matrix)


        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        '''
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_RF]
        for ind, fit in zip(invalid_ind, prediction_RF):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_LR]
        for ind, fit in zip(invalid_ind, prediction_LR):
            ind.fitness.values = fit
        '''



        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_NB]
        for ind, fit in zip(invalid_ind, prediction_NB):
            ind.fitness.values = fit

        # Estimation of the individuals fitness using simplified simulation
        '''
        # Half shop
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_halfshop], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_halfshop, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_halfshop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Single Replication
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRep, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Single Replication short
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep_short],
                               dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRepShort, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep_short]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        '''

        # Select the next generation individuals based on the prediction
        population_intermediate = toolbox.select(population_intermediate_predicted_NB, pop_size)
        print(f'population size of intermediate population after selection: {len(population_intermediate)}')

        # remove the estimated fitness from the population to add
        for ind in population_intermediate:
            del ind.fitness.values

        # add the pre-selected individuals to the population
        if gen < ngen:
            population += population_intermediate

        print(
            f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Update random seed
        rand_value = random.randint(1, 300)
        end = time.time()
        time_current_generation = end - start
        time_per_generation.append(time_current_generation)
        print(f'Execution time one loop: {time_current_generation}')

    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    time_dict = {'time per generation': time_per_generation, 'total time': total_time}
    return population, logbook, time_dict, population_dict, fitness_dict, fitness_evaluations


# GPHH with surrogates (using the latest 1000 fully evaluated individuals for training of surrogates)
# optimized surrogate evaluation -> only generate the phenotypic characterization for not yet characterized individuals)
def GPHH_experiment2_KNN(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref, n, n_offspring,
                        stats=None,
                        halloffame=None, verbose=__debug__):
    # Initialize count for total time
    start_total_time = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1, 300)

    # initialize the lists for measuring the time
    time_per_generation = []

    # Update random seed
    rand_value = random.randint(1, 300)

    # Initialize lists for populations
    population_dict = {}
    fitness_dict = {}

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population) / 2)
    pop_size_offspring = int(len(population))

    # Begin the generational process
    for gen in range(1, ngen + 1):
        start = time.time()
        print(f'Generation: {gen}')
        population = remove_duplicates(population=population)
        print(f'population size: {len(population)}')

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # save the full evaluated individuals to the population dict
        population_dict[f'Individuals Generation {gen}'] = [str(ind) for ind in population]

        fitness_list = [str(ind.fitness) for ind in population]
        tuple_list = [tuple(float(s) for s in x.strip("()").split(",")) for x in fitness_list]
        f1_list = [a_tuple[0] for a_tuple in tuple_list]
        f2_list = [a_tuple[1] for a_tuple in tuple_list]
        fitness_dict[f'Fitness f1 Generation {gen}'] = f1_list
        fitness_dict[f'Fitness f2 Generation {gen}'] = f2_list

        # add the full evaluated individuals to the training set
        pop_train += invalid_ind
        if len(pop_train) > 1000:
            pop_train = pop_train[-1000:]
        print(f'population size of the training population: {len(pop_train)}')

        # generate the phenotypic characterization of the training individuals
        invalid_ind = np.array(
            [[ind, decision_situations, ranking_vector_ref] for ind in pop_train if ind.phenotypic == None],
            dtype=object)
        invalid_ind_1 = [ind for ind in pop_train if ind.phenotypic == None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in pop_train]
        fitness_matrix = [ind.fitness.values for ind in pop_train]
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')

        # Update the surrogate models (train the predictors)

        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        '''
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)
        NB_model = NB_train(X=decision_matrix, y=fitness_matrix)
        LR_model = LR_train(X=decision_matrix, y=fitness_matrix)
        RF_model = RF_train(X=decision_matrix, y=fitness_matrix)
        '''
        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = len(population) * n
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]

        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:(len(population) * n)]
        print(f'population size of intermediate population: {len(population_intermediate)}')

        # generate the phenotypic characterization of the individuals in the intermediate population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate if
                                ind.phenotypic == None], dtype=object)
        invalid_ind_1 = [ind for ind in population_intermediate if ind.phenotypic == None]
        phenotypic_eval = toolbox.map(toolbox.decision_vector, invalid_ind)
        for ind, phenotypic in zip(invalid_ind_1, phenotypic_eval):
            ind.phenotypic = phenotypic
        decision_matrix = [ind.phenotypic for ind in population_intermediate]
        print(f'Individuals that have been transformed to phenotypic characterization: {len(invalid_ind_1)}')

        # Create a duplicate of the intermediate population for the prediction
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        '''
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_NB = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_LR = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_RF = [toolbox.clone(ind) for ind in population_intermediate]
        '''

        # Create a duplicate of the intermediate population for the estimation using simplified simulation

        # population_intermediate_predicted_halfshop = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_singlerep = [toolbox.clone(ind) for ind in population_intermediate]
        # population_intermediate_predicted_singlerep_short = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness using ML techniques

        prediction_KNN = predict(KNN_model, decision_matrix)
        '''
        prediction_MLP = predict(MLP_model, decision_matrix)
        prediction_DT = predict(DT_model, decision_matrix)
        prediction_SVM = predict(SVM_model, decision_matrix)
        prediction_NB = predict(NB_model, decision_matrix)
        prediction_LR = predict(LR_model, decision_matrix)
        prediction_RF = predict(RF_model, decision_matrix)
        '''

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        '''
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_NB]
        for ind, fit in zip(invalid_ind, prediction_NB):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_LR]
        for ind, fit in zip(invalid_ind, prediction_LR):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_RF]
        for ind, fit in zip(invalid_ind, prediction_RF):
            ind.fitness.values = fit
        '''

        # Estimation of the individuals fitness using simplified simulation
        '''
        # Half shop
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_halfshop], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_halfshop, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_halfshop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Single Replication
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRep, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Single Replication short
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep_short],
                               dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRepShort, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep_short]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        '''

        # Select the next generation individuals based on the prediction
        population_intermediate = toolbox.select(population_intermediate_predicted_KNN, pop_size)
        print(f'population size of intermediate population after selection: {len(population_intermediate)}')

        # remove the estimated fitness from the population to add
        for ind in population_intermediate:
            del ind.fitness.values

        # add the pre-selected individuals to the population
        if gen < ngen:
            population += population_intermediate

        print(
            f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Update random seed
        rand_value = random.randint(1, 300)
        end = time.time()
        time_current_generation = end - start
        time_per_generation.append(time_current_generation)
        print(f'Execution time one loop: {time_current_generation}')

    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    time_dict = {'time per generation': time_per_generation, 'total time': total_time}
    return population, logbook, time_dict, population_dict, fitness_dict


# GPHH without surrogates
def GPHH_experiment2_WS(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref,n , n_offspring, stats=None,
             halloffame=None, verbose=__debug__):

    # Initialize count for total time
    start_total_time = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)

    # initialize the lists for measuring the time
    time_per_generation = []

    # Update random seed
    rand_value = random.randint(1,300)

    # Initialize lists for populations
    population_dict = {}
    fitness_dict = {}

    # Initialize the population to train the classifier
    pop_train = []
    pop_size = int(len(population)/2)
    pop_size_offspring = int(len(population))

    # Begin the generational process
    for gen in range(1, ngen + 1):
        start = time.time()
        print(f'Generation: {gen}')
        population = remove_duplicates(population=population)
        print(f'population size: {len(population)}')

        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'Evaluated individuals: {len(invalid_ind)}')

        # save the full evaluated individuals to the population dict
        population_dict[f'Individuals Generation {gen}'] = [str(ind) for ind in population]

        fitness_list = [str(ind.fitness) for ind in population]
        tuple_list = [tuple(float(s) for s in x.strip("()").split(",")) for x in fitness_list]
        f1_list = [a_tuple[0] for a_tuple in tuple_list]
        f2_list = [a_tuple[1] for a_tuple in tuple_list]
        fitness_dict[f'Fitness f1 Generation {gen}'] = f1_list
        fitness_dict[f'Fitness f2 Generation {gen}'] = f2_list

        # Select the next generation individuals based on the evaluation
        population = toolbox.select(population, pop_size)
        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_size_intermediate = len(population)
        population_intermediate = []
        while len(population_intermediate) < population_size_intermediate:
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring
            population_intermediate = remove_duplicates(population=population_intermediate)
            population_intermediate = [ind for ind in population_intermediate if not ind.fitness.valid]

        # cut the intermediate population to its defined size
        population_intermediate = population_intermediate[:(len(population))]
        print(f'population size of intermediate population: {len(population_intermediate)}')


        # add the selected individuals to the population
        if gen < ngen:
            population += population_intermediate

        print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Update random seed
        rand_value = random.randint(1, 300)
        end = time.time()
        time_current_generation = end-start
        time_per_generation.append(time_current_generation)
        print(f'Execution time one loop: {time_current_generation}')

    end_total_time = time.time()
    total_time = end_total_time-start_total_time
    time_dict = {'time per generation': time_per_generation, 'total time': total_time}
    return population, logbook, time_dict, population_dict, fitness_dict



# GPHH for testing the selection accuracy (using only the last two generations for the training of the surrogates)
def GPHH_experiment2_old(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref,n , n_offspring, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)
    '''
    # Evaluate the individuals with an invalid fitness using full evaluation
    invalid_ind = np.array([[ind, rand_value] for ind in population if not ind.fitness.valid], dtype=object)
    invalid_ind_1 = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind_1, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
'''
    # Update random seed
    rand_value = random.randint(1,300)

    # Initialize selection accuracy
    selection_accuracy_list = []
    selection_accuracy_list_simple = []

    # Initialize the population to train the classifier
    pop_train = []
    pop_size_offspring = int(len(population))
    pop_size = int(len(population)/2)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        start = time.time()
        print(f'population size: {len(population)}')
        # Evaluate the individuals with an invalid fitness using full evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'Evaluated individuals: {len(invalid_ind)}')

        pop_train += invalid_ind
        print(pop_train)

        print(f'population size of the training population: {len(pop_train)}')

        # generate the phenotypic characterization of the individuals
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in pop_train], dtype=object)
        invalid_ind_1 = [ind for ind in pop_train]
        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = np.array(decision_matrix)
        fitness_matrix = [ind.fitness.values for ind in pop_train]

        # Update the surrogate models (train the predictors)
        KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        MLP_model = MLP_train(X=decision_matrix, y=fitness_matrix)
        DT_model = DT_train(X=decision_matrix, y=fitness_matrix)
        SVM_model = SVM_train(X=decision_matrix, y=fitness_matrix)

        # Select the next generation individuals based on the evaluation
        pop_size_training = int(2*len(population))
        population = toolbox.select(population, pop_size)

        print(f'population size after selection: {len(population)}')

        # produce offsprings
        population_intermediate = []
        for i in range(n):
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            population_intermediate += offspring

        print(f'population size of intermediate population: {len(population_intermediate)}')

        # Evaluate the individuals with an invalid fitness using full evaluation (only for comparison purpose to calculate the accuracy)
        invalid_ind = [ind for ind in population_intermediate if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.full_evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print(f'evaluated individuals of intermediate population: {len(invalid_ind)}')
        # add the full evaluated individual to the training population
        pop_train += invalid_ind

        # remove the oldest individuals after generation 2
        if gen >= 2:
            pop_train = pop_train[-pop_size_training:]

        # Assign a number to each individual in the intermediate population
        for i in range(len(population_intermediate)):
            population_intermediate[i].number = i

        # generate the phenotypic characterization of the individuals in the intermediate population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate], dtype=object)
        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = np.array(decision_matrix)

        # Create a duplicate of the intermediate population for the prediction
        population_intermediate_predicted_KNN = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_MLP = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_DT = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_SVM = [toolbox.clone(ind) for ind in population_intermediate]

        # Create a duplicate of the intermediate population for the estimation using simplified simulation
        population_intermediate_predicted_halfshop = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_singlerep = [toolbox.clone(ind) for ind in population_intermediate]
        population_intermediate_predicted_singlerep_short = [toolbox.clone(ind) for ind in population_intermediate]

        # Estimation of the individuals fitness using ML techniques
        prediction_KNN = predict(KNN_model, decision_matrix)
        prediction_MLP = predict(MLP_model, decision_matrix)
        prediction_DT = predict(DT_model, decision_matrix)
        prediction_SVM = predict(SVM_model, decision_matrix)

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_KNN]
        for ind, fit in zip(invalid_ind, prediction_KNN):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_MLP]
        for ind, fit in zip(invalid_ind, prediction_MLP):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_DT]
        for ind, fit in zip(invalid_ind, prediction_DT):
            ind.fitness.values = fit
        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        invalid_ind = [ind for ind in population_intermediate_predicted_SVM]
        for ind, fit in zip(invalid_ind, prediction_SVM):
            ind.fitness.values = fit

        # Estimation of the individuals fitness using simplified simulation
        # Half shop
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_halfshop], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_halfshop, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_halfshop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Single Replication
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep], dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRep, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Single Replication short
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate_predicted_singlerep_short],
                               dtype=object)
        fitnesses = toolbox.map(toolbox.evaluate_singleRepShort, invalid_ind)
        invalid_ind = [ind for ind in population_intermediate_predicted_singlerep_short]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation individuals based on the evaluation
        population_intermediate = toolbox.select(population_intermediate, pop_size)
        print(f'population size of intermediate population after selection: {len(population_intermediate)}')

        # Select the next generation individuals based on the estimation (just for evaluation purposes, no real selection is made)
        population_predicted_KNN = toolbox.select(population_intermediate_predicted_KNN, pop_size)
        population_predicted_MLP = toolbox.select(population_intermediate_predicted_MLP, pop_size)
        population_predicted_DT = toolbox.select(population_intermediate_predicted_DT, pop_size)
        population_predicted_SVM = toolbox.select(population_intermediate_predicted_SVM, pop_size)

        # Select the next generation individuals based on the simplified simulations
        population_predicted_halfshop = toolbox.select(population_intermediate_predicted_halfshop, pop_size)
        population_predicted_singerep= toolbox.select(population_intermediate_predicted_singlerep, pop_size)
        population_predicted_singerep_short= toolbox.select(population_intermediate_predicted_singlerep_short, pop_size)

        # compare both selections and calculate the selection accuracy
        ranking = []
        ranking_predicted_KNN = []
        ranking_predicted_MLP = []
        ranking_predicted_DT = []
        ranking_predicted_SVM = []
        ranking_predicted_halfshop = []
        ranking_predicted_singlerep = []
        ranking_predicted_singlerep_short = []
        for i in population_intermediate:
            ranking.append(i.number)
        for i in population_predicted_KNN:
            ranking_predicted_KNN.append(i.number)
        for i in population_predicted_MLP:
            ranking_predicted_MLP.append(i.number)
        for i in population_predicted_DT:
            ranking_predicted_DT.append(i.number)
        for i in population_predicted_SVM:
            ranking_predicted_SVM.append(i.number)
        for i in population_predicted_halfshop:
            ranking_predicted_halfshop.append(i.number)
        for i in population_predicted_singerep:
            ranking_predicted_singlerep.append(i.number)
        for i in population_predicted_singerep_short:
            ranking_predicted_singlerep_short.append(i.number)
        print(f'Ranking actual: {ranking}')
        print(f'Ranking KNN: {ranking_predicted_KNN}')
        print(f'Ranking MLP: {ranking_predicted_MLP}')
        print(f'Ranking DT: {ranking_predicted_DT}')
        print(f'Ranking SVM: {ranking_predicted_SVM}')
        print(f'Ranking halfshop: {ranking_predicted_halfshop}')
        print(f'Ranking single replication: {ranking_predicted_singlerep}')
        print(f'Ranking single replication short: {ranking_predicted_singlerep_short}')
        selection_accuracy_KNN, selection_accuracy_KNN_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_KNN,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_MLP, selection_accuracy_MLP_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_MLP,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_DT, selection_accuracy_DT_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_DT,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_SVM, selection_accuracy_SVM_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_SVM,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_halfshop, selection_accuracy_halfshop_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_halfshop,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_singlerep, selection_accuracy_singlerep_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_singlerep,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)
        selection_accuracy_singlerep_short, selection_accuracy_singlerep_short_simple = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted_singlerep_short,
                                                         pop_size_offspring=pop_size_offspring, pop_size_parents=pop_size)

        selection_accuracy_list.append([selection_accuracy_KNN, selection_accuracy_MLP, selection_accuracy_DT, selection_accuracy_SVM, selection_accuracy_halfshop, selection_accuracy_singlerep, selection_accuracy_singlerep_short])
        selection_accuracy_list_simple.append([selection_accuracy_KNN_simple, selection_accuracy_MLP_simple, selection_accuracy_DT_simple, selection_accuracy_SVM_simple, selection_accuracy_halfshop_simple, selection_accuracy_singlerep_simple, selection_accuracy_singlerep_short_simple])

        print('selection accuracy ')
        print(selection_accuracy_list)
        print('selection accuracy simple')
        print(selection_accuracy_list_simple)

        # remove the estimated fitness from the population to add
        #for ind in population_intermediate:
        #    del ind.fitness.values

        # add the pre-selected individuals to the population
        population += population_intermediate

        print(f'population size of population after adding the selected individuals from the intermediate population: {len(population)}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Append the current generation statistics to the logbook
        #record = stats.compile(population) if stats else {}
        #logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        #if verbose:
        #    print(logbook.stream)

        # Update random seed
        rand_value = random.randint(1, 300)
        end = time.time()
        print(f'Execution time one loop: {end - start}')
    return population, logbook, selection_accuracy_list, selection_accuracy_list_simple

# eaCustom (including changing random seed for each generation)
def eaCustom(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = np.array([[ind, rand_value] for ind in population if not ind.fitness.valid])
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    invalid_ind = list(invalid_ind[:,0])
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Update random seed
    rand_value = random.randint(1,300)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = np.array([[ind, rand_value] for ind in offspring if not ind.fitness.valid])
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        invalid_ind = list(invalid_ind[:, 0])
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Update random seed
        rand_value = random.randint(1, 300)

    return population, logbook

# eaCustom with decision vector
def eaCustom_dv(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    final_decision_matrix = []

    # set random seed
    rand_value = random.randint(1,300)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = np.array([[ind, rand_value] for ind in population if not ind.fitness.valid])
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    invalid_ind = list(invalid_ind[:,0])
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
    decision_matrix = [decision_matrix[i] + [fitnesses[i]] for i in range(len(invalid_ind))]
    final_decision_matrix.extend(decision_matrix)
    #print(len(final_decision_matrix))
    #print(final_decision_matrix)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Update random seed
    rand_value = random.randint(1,300)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = np.array([[ind, rand_value] for ind in offspring if not ind.fitness.valid])
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        invalid_ind = list(invalid_ind[:, 0])
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = [decision_matrix[i] + [fitnesses[i]] for i in range(len(invalid_ind))]
        final_decision_matrix.extend(decision_matrix)
        #print(len(final_decision_matrix))
        #print(final_decision_matrix)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Update random seed
        rand_value = random.randint(1, 300)

    return population, logbook, final_decision_matrix

# surrogate-assisted GP-HH
def GPHH_Surrogate(population, toolbox, cxpb, mutpb, ngen, decision_situations, ranking_vector_ref, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # set random seed
    rand_value = random.randint(1,300)

    # create a copy of the population for the prediction
    #population_predicted = population

    # Evaluate the individuals with an invalid fitness using full evaluation
    invalid_ind = np.array([[ind, rand_value] for ind in population if not ind.fitness.valid])
    invalid_ind_1 = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind_1, fitnesses):
        ind.fitness.values = fit

    # generate the phenotypic characterization of the individuals
    invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population])
    decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
    decision_matrix = np.array(decision_matrix)
    #print(decision_matrix)
    invalid_ind = list(invalid_ind[:, 0])
    fitness_matrix = [fitnesses[i] for i in range(len(invalid_ind))]
    #print(fitness_matrix)
    #decision_matrix = np.array([decision_matrix[i] + [fitnesses[i]] for i in range(len(invalid_ind))])
    #phenotypic_characterization = decision_matrix
    #print(phenotypic_characterization)
    #print(len(final_decision_matrix))
    #print(final_decision_matrix)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Update random seed
    rand_value = random.randint(1,300)

    # Update the surrogate model (train the predictors)
    KNN_model = KNN_train(X=decision_matrix, y=fitness_matrix)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        print(f'Generation: {gen}')

        # produce offsprings
        offspring = varAnd(population, toolbox, cxpb, mutpb)

        # combine population and offspring to a new intermediate population
        offspring.extend(population)
        population_intermediate = offspring
        #print(f'intermediate population: {population_intermediate}')

        # Evaluate the individuals with an invalid fitness
        invalid_ind = np.array([[ind, rand_value] for ind in population_intermediate if not ind.fitness.valid])
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        invalid_ind = list(invalid_ind[:, 0])
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Assign a number to each individual in the intermediate population
        #for i in range(len(population_intermediate)):
        #    population_intermediate[i].number = i

        # Create a duplicate of the intermediate population for the prediction
        population_intermediate_predicted = population_intermediate

        # generate the phenotypic characterization of the individuals in the intermediate population
        invalid_ind = np.array([[ind, decision_situations, ranking_vector_ref] for ind in population_intermediate_predicted])
        decision_matrix = toolbox.map(toolbox.decision_vector, invalid_ind)
        decision_matrix = np.array(decision_matrix)
        print(decision_matrix)
        print(len(decision_matrix))
        #fitness_matrix = [fitnesses[i] for i in range(len(invalid_ind))]
        #print(fitness_matrix)

        # Estimation of the individuals fitness
        prediction = predict(KNN_model, decision_matrix)
        print(prediction)

        #for i in range(len(population_intermediate)):
        #    print(f'fitness of {i}th individual: {population_intermediate[i].fitness}')

        # Assign predicted fitness to the individuals of the intermediate population in a new population (predicted)
        for i in range(len(population_intermediate_predicted)):
            population_intermediate_predicted[i].fitness = prediction[i]

        #invalid_ind = [ind for ind in population_intermediate_predicted]
        #for ind, fit in zip(invalid_ind, prediction):
        #    ind.fitness.values = fit

        #for i in range(len(population_intermediate_predicted)):
        #    print(f'fitness of {i}th individual: {population_intermediate_predicted[i].fitness}')

        # Select the next generation individuals based on the evaluation
        print(population_intermediate[0].fitness)
        #print(population_intermediate[0].number)
        population = toolbox.select(population_intermediate, len(population))

        # Select the next generation individuals based on the estimation
        population_predicted = toolbox.select(population_intermediate_predicted, pop_size)

        # compare both selections and calculate the selection accuracy
        ranking = []
        ranking_predicted = []
        for i in population:
            ranking.append(i.number)
        for i in population_predicted:
            ranking_predicted.append(i.number)
        print(ranking)
        print(ranking_predicted)
        selection_accuracy = evaluate_selection_accuracy(ranking_actual=ranking, ranking_predicted=ranking_predicted,
                                                         pop_size_offspring=len(population_intermediate), pop_size_parents=len(population))
        print(f'selection accuracy: {selection_accuracy}')

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Update random seed
        rand_value = random.randint(1, 300)

    return population, logbook





def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population.

    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.

    This variation is named *Or* because an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.

    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)

    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu~,~\lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(offspring, mu)

    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from **only** the offspring. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Care must be taken when the lambda:mu ratio is 1 to 1 as a
        non-stochastic selection will result in no selection at all as the
        operator selects *lambda* individuals from a pool of *mu*.


    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    return population, logbook


def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__):
    """This is algorithm implements the ask-tell model proposed in
    [Colette2010]_, where ask is called `generate` and tell is called `update`.

    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm generates the individuals using the :func:`toolbox.generate`
    function and updates the generation method with the :func:`toolbox.update`
    function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The pseudocode goes as follow ::

        for g in range(ngen):
            population = toolbox.generate()
            evaluate(population)
            toolbox.update(population)

    .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
       R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
       Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
       Multidisciplinary Design Optimization in Computational Mechanics,
       Wiley, pp. 527-565;

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

from GPHH import main
import time
import pandas as pd


if __name__ == '__main__':
    for i in range(1, 31):
        start = time.time()
        main(run=i)
        end = time.time()
        print(f'Execution time simulation: {end - start}')
    #print(f'Best solution: {best_solution}')
    #print(f'Best Fitness: {best_fitness}')
    #print(minFitnessValues)

    #df_minFitnessValues = pd.DataFrame(minFitnessValues)
    #df_minFitnessValues.to_excel("output.xlsx")

    #df_solution.to_excel("solution.xlsx")

    #df_log = pd.DataFrame(
    #    {'Number Generation': nb_generation,
    #     'Average Fitness': avgFitnessValues,
    #     'Min Fitness': minFitnessValues,
    #     'Max Fitness': maxFitnessValues,
    #     'Std Fitness': stdFitnessValues
    #     })
    #df_log.to_excel("log.xlsx")
import pygmo as pg
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean

hv_list_WS_average = []
hv_list_RF_average = []
hv_list_SR_average = []
hv_list_DT_average = []
hv_list_NB_average = []

for j in range(1,31):
    print(f'Run: {j}')

    # WS
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/WS/run_{str(j)}/fitness.xlsx'
    fitness_WS = pd.read_excel(link, header=0, index_col=0)
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/WS/run_{str(j)}/fitness_evaluations.xlsx'
    fitness_evaluations_WS = pd.read_excel(link, header=0, index_col=0)

    #RF
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/RF_duplicate/run_{str(j)}/fitness.xlsx'
    fitness_RF = pd.read_excel(link, header=0, index_col=0)
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/RF_duplicate/run_{str(j)}/fitness_evaluations.xlsx'
    fitness_evaluations_RF = pd.read_excel(link, header=0, index_col=0)

    #SR
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/SR/run_{str(j)}/fitness.xlsx'
    fitness_SR = pd.read_excel(link, header=0, index_col=0)
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/SR/run_{str(j)}/fitness_evaluations.xlsx'
    fitness_evaluations_SR = pd.read_excel(link, header=0, index_col=0)

    #DT
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/DT/run_{str(j)}/fitness.xlsx'
    fitness_DT = pd.read_excel(link, header=0, index_col=0)
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/DT/run_{str(j)}/fitness_evaluations.xlsx'
    fitness_evaluations_DT = pd.read_excel(link, header=0, index_col=0)

    #NB
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/NB/run_{str(j)}/fitness.xlsx'
    fitness_NB = pd.read_excel(link, header=0, index_col=0)
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/NB/run_{str(j)}/fitness_evaluations.xlsx'
    fitness_evaluations_NB = pd.read_excel(link, header=0, index_col=0)



    f1_dict_WS = []
    f2_dict_WS = []

    f1_dict_RF = []
    f2_dict_RF = []

    f1_dict_SR = []
    f2_dict_SR = []

    f1_dict_DT = []
    f2_dict_DT = []

    f1_dict_NB = []
    f2_dict_NB = []

    hv_list_WS = []
    hv_list_RF = []
    hv_list_SR = []
    hv_list_DT = []
    hv_list_NB = []

    evaluations_list_WS = []
    evaluations_list_RF = []
    evaluations_list_SR = []
    evaluations_list_DT = []
    evaluations_list_NB = []

    for i in range(1, 51):
        # drop na
        # WS
        f1_WS = fitness_WS[f'Fitness f1 Generation {str(i)}']
        f1_WS = f1_WS.dropna()
        f2_WS = fitness_WS[f'Fitness f2 Generation {str(i)}']
        f2_WS = f2_WS.dropna()
        # SR
        f1_SR = fitness_SR[f'Fitness f1 Generation {str(i)}']
        f1_SR = f1_SR.dropna()
        f2_SR = fitness_SR[f'Fitness f2 Generation {str(i)}']
        f2_SR = f2_SR.dropna()

        # calculate fitness evaluations
        fitness_evaluations_WS_count = fitness_evaluations_WS.iloc[:i, 0]
        fitness_evaluations_WS_count = fitness_evaluations_WS_count.values.tolist()
        fitness_evaluations_WS_count = sum(fitness_evaluations_WS_count) * 5
        evaluations_list_WS.append(fitness_evaluations_WS_count)
        fitness_evaluations_SR_count = fitness_evaluations_SR.iloc[:i, 0]
        fitness_evaluations_SR_count = fitness_evaluations_SR_count.values.tolist()
        fitness_evaluations_SR_count = sum(fitness_evaluations_SR_count)
        evaluations_list_SR.append(fitness_evaluations_SR_count)

        f1_dict_WS.append(f1_WS)
        f2_dict_WS.append(f2_WS)
        f1_dict_SR.append(f1_SR)
        f2_dict_SR.append(f2_SR)


    # RF (140 generations)
    for i in range(1, 141):
        # drop na
        #RF
        f1_RF = fitness_RF[f'Fitness f1 Generation {str(i)}']
        f1_RF = f1_RF.dropna()
        f2_RF = fitness_RF[f'Fitness f2 Generation {str(i)}']
        f2_RF = f2_RF.dropna()
        #DT
        f1_DT = fitness_DT[f'Fitness f1 Generation {str(i)}']
        f1_DT = f1_DT.dropna()
        f2_DT = fitness_DT[f'Fitness f2 Generation {str(i)}']
        f2_DT = f2_DT.dropna()
        #NB
        f1_NB = fitness_NB[f'Fitness f1 Generation {str(i)}']
        f1_NB = f1_NB.dropna()
        f2_NB = fitness_NB[f'Fitness f2 Generation {str(i)}']
        f2_NB = f2_NB.dropna()

        # calculate fitness evaluations
        #RF
        fitness_evaluations_RF_count = fitness_evaluations_RF.iloc[:i, 0]
        fitness_evaluations_RF_count = fitness_evaluations_RF_count.values.tolist()
        fitness_evaluations_RF_count = sum(fitness_evaluations_RF_count) * 5
        evaluations_list_RF.append(fitness_evaluations_RF_count)
        f1_dict_RF.append(f1_RF)
        f2_dict_RF.append(f2_RF)
        #DT
        fitness_evaluations_DT_count = fitness_evaluations_DT.iloc[:i, 0]
        fitness_evaluations_DT_count = fitness_evaluations_DT_count.values.tolist()
        fitness_evaluations_DT_count = sum(fitness_evaluations_DT_count) * 5
        evaluations_list_DT.append(fitness_evaluations_DT_count)
        f1_dict_DT.append(f1_DT)
        f2_dict_DT.append(f2_DT)
        #NB
        fitness_evaluations_NB_count = fitness_evaluations_NB.iloc[:i, 0]
        fitness_evaluations_NB_count = fitness_evaluations_NB_count.values.tolist()
        fitness_evaluations_NB_count = sum(fitness_evaluations_NB_count) * 5
        evaluations_list_NB.append(fitness_evaluations_NB_count)
        f1_dict_NB.append(f1_NB)
        f2_dict_NB.append(f2_NB)

    # find the indexes of each list where the replications are first more than 5000
    index_WS = next(x for x, val in enumerate(evaluations_list_WS)
               if val >= 2500)
    index_SR = next(x for x, val in enumerate(evaluations_list_SR)
               if val >= 2500)
    index_RF = next(x for x, val in enumerate(evaluations_list_RF)
               if val >= 2500)
    index_DT = next(x for x, val in enumerate(evaluations_list_DT)
               if val >= 2500)
    index_NB = next(x for x, val in enumerate(evaluations_list_NB)
               if val >= 2500)

    print(index_WS)
    print(index_SR)
    print(index_RF)
    print(index_DT)
    print(index_NB)

    fig, axs = plt.subplots(5, 3)
    fig.set_size_inches(11, 14, forward=True)
    #fig.suptitle(f'MO-GP-HH using Random Forest as surrogate and with improved duplication removing (Run {j})')
    axs[0, 0].plot(f1_dict_WS[index_WS], f2_dict_WS[index_WS], 'o',  color="blue")
    axs[0, 0].set_xlim(left=350, right=500)
    axs[0, 0].set_ylim(bottom=0, top=3500)
    axs[0, 0].set(ylabel='Max Tardiness')
    #axs[0, 0].set_title('Simulation Replications ~5,000')
    axs[1, 0].plot(f1_dict_SR[index_SR], f2_dict_SR[index_SR], 'o',  color="orange")
    axs[1, 0].set_xlim(left=350, right=500)
    axs[1, 0].set_ylim(bottom=0, top=3500)
    axs[1, 0].set(ylabel='Max Tardiness')
    #axs[0, 1].set_title('Simulation Replications ~5,000')
    axs[2, 0].plot(f1_dict_RF[index_RF], f2_dict_RF[index_RF], 'o',  color="green")
    axs[2, 0].set_xlim(left=350, right=500)
    axs[2, 0].set_ylim(bottom=0, top=3500)
    axs[2, 0].set(ylabel='Max Tardiness')
    #axs[0, 2].set_title('Simulation Replications ~5,000')
    axs[3, 0].plot(f1_dict_DT[index_DT], f2_dict_DT[index_DT], 'o',  color="red")
    axs[3, 0].set_xlim(left=350, right=500)
    axs[3, 0].set_ylim(bottom=0, top=3500)
    axs[3, 0].set(ylabel='Max Tardiness')
    axs[4, 0].plot(f1_dict_NB[index_NB], f2_dict_NB[index_NB], 'o',  color="purple")
    axs[4, 0].set_xlim(left=350, right=500)
    axs[4, 0].set_ylim(bottom=0, top=3500)
    axs[4, 0].set(ylabel='Max Tardiness')
    axs[4, 0].set(xlabel='Mean Flowtime')

    # find the indexes of each list where the replications are first more than 10000
    index_WS = next(x for x, val in enumerate(evaluations_list_WS)
               if val >= 10000)
    index_SR = next(x for x, val in enumerate(evaluations_list_SR)
               if val >= 10000)
    index_RF = next(x for x, val in enumerate(evaluations_list_RF)
               if val >= 10000)
    index_DT = next(x for x, val in enumerate(evaluations_list_DT)
               if val >= 10000)
    index_NB = next(x for x, val in enumerate(evaluations_list_NB)
               if val >= 10000)

    print(index_WS)
    print(index_SR)
    print(index_RF)

    axs[0, 1].plot(f1_dict_WS[index_WS], f2_dict_WS[index_WS], 'o',  color="blue")
    axs[0, 1].set_xlim(left=350, right=500)
    axs[0, 1].set_ylim(bottom=0, top=3500)
    #axs[1, 0].set_title('Simulation Replications ~10,000')
    axs[1, 1].plot(f1_dict_SR[index_SR], f2_dict_SR[index_SR], 'o',  color="orange")
    axs[1, 1].set_xlim(left=350, right=500)
    axs[1, 1].set_ylim(bottom=0, top=3500)
    #axs[1, 1].set_title('Simulation Replications ~10,000')
    axs[2, 1].plot(f1_dict_RF[index_RF], f2_dict_RF[index_RF], 'o',  color="green")
    axs[2, 1].set_xlim(left=350, right=500)
    axs[2, 1].set_ylim(bottom=0, top=3500)
    #axs[1, 2].set_title('Simulation Replications ~10,000')
    axs[3, 1].plot(f1_dict_DT[index_DT], f2_dict_DT[index_DT], 'o',  color="red")
    axs[3, 1].set_xlim(left=350, right=500)
    axs[3, 1].set_ylim(bottom=0, top=3500)
    axs[4, 1].plot(f1_dict_NB[index_NB], f2_dict_NB[index_NB], 'o',  color="purple")
    axs[4, 1].set_xlim(left=350, right=500)
    axs[4, 1].set_ylim(bottom=0, top=3500)
    axs[4, 1].set(xlabel='Mean Flowtime')

    # find the indexes of each list where the replications are first more than 10000
    index_WS = next(x for x, val in enumerate(evaluations_list_WS)
               if val >= 25000)
    index_SR = next(x for x, val in enumerate(evaluations_list_SR)
               if val >= 25000)
    try:
        index_RF = next(x for x, val in enumerate(evaluations_list_RF)
                if val >= 25000)
    except:
        index_RF = 139

    try:
        index_DT = next(x for x, val in enumerate(evaluations_list_DT)
                if val >= 25000)
    except:
        index_DT = 139

    try:
        index_NB = next(x for x, val in enumerate(evaluations_list_NB)
                if val >= 25000)
    except:
        index_NB = 139

    print(index_WS)
    print(index_SR)
    print(index_RF)

    axs[0, 2].plot(f1_dict_WS[index_WS], f2_dict_WS[index_WS], 'o',  color="blue")
    axs[0, 2].set_xlim(left=350, right=500)
    axs[0, 2].set_ylim(bottom=0, top=3500)
    #axs[2, 0].set_title('Simulation Replications ~25,000')
    axs[1, 2].plot(f1_dict_SR[index_SR], f2_dict_SR[index_SR], 'o',  color="orange")
    axs[1, 2].set_xlim(left=350, right=500)
    axs[1, 2].set_ylim(bottom=0, top=3500)
    #axs[2, 1].set_title('Simulation Replications ~25,000')
    axs[2, 2].plot(f1_dict_RF[index_RF], f2_dict_RF[index_RF], 'o',  color="green")
    axs[2, 2].set_xlim(left=350, right=500)
    axs[2, 2].set_ylim(bottom=0, top=3500)
    #axs[2, 2].set_title('Simulation Replications ~25,000')
    axs[3, 2].plot(f1_dict_DT[index_DT], f2_dict_DT[index_DT], 'o',  color="red")
    axs[3, 2].set_xlim(left=350, right=500)
    axs[3, 2].set_ylim(bottom=0, top=3500)
    axs[4, 2].plot(f1_dict_NB[index_NB], f2_dict_NB[index_NB], 'o',  color="purple")
    axs[4, 2].set_xlim(left=350, right=500)
    axs[4, 2].set_ylim(bottom=0, top=3500)
    axs[4, 2].set(xlabel='Mean Flowtime')

    #for ax in axs.flat:
    #    ax.set(xlabel='mean flowtime', ylabel='max tardiness')
    plt.tight_layout()
    plt.savefig(f'population_run{j}.png', dpi=600)
    #plt.show()
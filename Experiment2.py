import pygmo as pg
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from scipy.stats import wilcoxon
from numpy import mean, std

hv_list_WS_average = []
hv_list_RF_average = []
hv_list_SR_average = []
hv_list_DT_average = []
hv_list_NB_average = []

hv_dict_WS = []
hv_dict_RF = []
hv_dict_SR = []
hv_dict_DT = []
hv_dict_NB = []

evaluations_dict_WS = []
evaluations_dict_SR = []
evaluations_dict_RF = []
evaluations_dict_DT = []
evaluations_dict_NB = []

f1f2_reference = []

for j in range(1,31):
    print(f'Run: {j}')

    # load files
    # WS
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/WS/run_{str(j)}/fitness.xlsx'
    fitness_WS = pd.read_excel(link, header=0, index_col=0)
    # RF
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/RF_duplicate/run_{str(j)}/fitness.xlsx'
    fitness_RF = pd.read_excel(link, header=0, index_col=0)
    #SR
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/SR/run_{str(j)}/fitness.xlsx'
    fitness_SR = pd.read_excel(link, header=0, index_col=0)
    #DT
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/DT/run_{str(j)}/fitness.xlsx'
    fitness_DT = pd.read_excel(link, header=0, index_col=0)
    #NB
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/NB/run_{str(j)}/fitness.xlsx'
    fitness_NB = pd.read_excel(link, header=0, index_col=0)


    # calculate the hypervolume for each run for each generation
    # WS & SR (50 generations)
    for i in range(1, 51):
        # drop na
        f1f2_WS = fitness_WS[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_WS = f1f2_WS.dropna()
        f1f2_SR = fitness_SR[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_SR = f1f2_SR.dropna()

        f1f2_WS = f1f2_WS.values.tolist()
        f1f2_SR = f1f2_SR.values.tolist()

        f1f2_reference.extend(f1f2_WS)
        f1f2_reference.extend(f1f2_SR)


    # RF and DT and NB (140 generations)
    for i in range(1, 141):
        # RF
        # drop na
        f1f2_RF = fitness_RF[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_RF = f1f2_RF.dropna()
        f1f2_RF = f1f2_RF.values.tolist()
        f1f2_reference.extend(f1f2_RF)

        # DT
        # drop na
        f1f2_DT = fitness_DT[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_DT = f1f2_DT.dropna()
        f1f2_DT = f1f2_DT.values.tolist()
        f1f2_reference.extend(f1f2_DT)

        # NB
        # drop na
        f1f2_NB = fitness_NB[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_NB = f1f2_NB.dropna()
        f1f2_NB = f1f2_NB.values.tolist()
        f1f2_reference.extend(f1f2_NB)

f1_ref = [item[0] for item in f1f2_reference]
f2_ref = [item[1] for item in f1f2_reference]

roh = 0.000001
f1_nadir = max(f1_ref) + roh
f2_nadir = max(f2_ref) + roh

print(f1_nadir)
print(f2_nadir)

# calculate HV reference
hyp_ref = pg.hypervolume(f1f2_reference)
hv_ref = hyp_ref.compute([f1_nadir, f2_nadir])

print(hv_ref)

hv_dict_WS = []
hv_dict_RF = []
hv_dict_SR = []
hv_dict_DT = []
hv_dict_NB = []

evaluations_dict_WS = []
evaluations_dict_SR = []
evaluations_dict_RF = []
evaluations_dict_DT = []
evaluations_dict_NB = []

for j in range(1,31):
    print(f'Run: {j}')

    # load files
    # WS
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/WS/run_{str(j)}/fitness.xlsx'
    fitness_WS = pd.read_excel(link, header=0, index_col=0)
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/WS/run_{str(j)}/fitness_evaluations.xlsx'
    fitness_evaluations_WS = pd.read_excel(link, header=0, index_col=0)
    # RF
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

    # Initialize lists
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

    # calculate the hypervolume for each run for each generation
    # WS & SR (50 generations)
    for i in range(1, 51):
        # drop na
        f1f2_WS = fitness_WS[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_WS = f1f2_WS.dropna()
        f1f2_SR = fitness_SR[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_SR = f1f2_SR.dropna()

        # calculate hypervolume
        hyp_WS = pg.hypervolume(f1f2_WS.values)
        hv_WS = hyp_WS.compute([f1_nadir, f2_nadir])
        hv_WS = hv_WS/hv_ref
        hv_list_WS.append(hv_WS)
        hyp_SR = pg.hypervolume(f1f2_SR.values)
        hv_SR = hyp_SR.compute([f1_nadir, f2_nadir])
        hv_SR = hv_SR/hv_ref
        hv_list_SR.append(hv_SR)

        # calculate fitness evaluations
        fitness_evaluations_WS_count = fitness_evaluations_WS.iloc[:i, 0]
        fitness_evaluations_WS_count = fitness_evaluations_WS_count.values.tolist()
        fitness_evaluations_WS_count = sum(fitness_evaluations_WS_count) * 5
        evaluations_list_WS.append(fitness_evaluations_WS_count)
        fitness_evaluations_SR_count = fitness_evaluations_SR.iloc[:i, 0]
        fitness_evaluations_SR_count = fitness_evaluations_SR_count.values.tolist()
        fitness_evaluations_SR_count = sum(fitness_evaluations_SR_count)
        evaluations_list_SR.append(fitness_evaluations_SR_count)

    # RF & DT & NB (140 generations)
    for i in range(1, 141):
        # drop na
        # RF
        f1f2_RF = fitness_RF[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_RF = f1f2_RF.dropna()
        # DT
        f1f2_DT = fitness_DT[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_DT = f1f2_DT.dropna()
        # NB
        f1f2_NB = fitness_NB[[f'Fitness f1 Generation {str(i)}',f'Fitness f2 Generation {str(i)}']]
        f1f2_NB = f1f2_NB.dropna()

        # calculate hypervolume
        # RF
        hyp_RF = pg.hypervolume(f1f2_RF.values)
        hv_RF = hyp_RF.compute([f1_nadir, f2_nadir])
        hv_RF = hv_RF/hv_ref
        hv_list_RF.append(hv_RF)
        # DT
        hyp_DT = pg.hypervolume(f1f2_DT.values)
        hv_DT = hyp_DT.compute([f1_nadir, f2_nadir])
        hv_DT = hv_DT/hv_ref
        hv_list_DT.append(hv_DT)
        # NB
        hyp_NB = pg.hypervolume(f1f2_NB.values)
        hv_NB = hyp_NB.compute([f1_nadir, f2_nadir])
        hv_NB = hv_NB/hv_ref
        hv_list_NB.append(hv_NB)

        # calculate fitness evaluations
        # RF
        fitness_evaluations_RF_count = fitness_evaluations_RF.iloc[:i, 0]
        fitness_evaluations_RF_count = fitness_evaluations_RF_count.values.tolist()
        fitness_evaluations_RF_count = sum(fitness_evaluations_RF_count) * 5
        evaluations_list_RF.append(fitness_evaluations_RF_count)
        # DT
        fitness_evaluations_DT_count = fitness_evaluations_DT.iloc[:i, 0]
        fitness_evaluations_DT_count = fitness_evaluations_DT_count.values.tolist()
        fitness_evaluations_DT_count = sum(fitness_evaluations_DT_count) * 5
        evaluations_list_DT.append(fitness_evaluations_DT_count)
        # NB
        fitness_evaluations_NB_count = fitness_evaluations_NB.iloc[:i, 0]
        fitness_evaluations_NB_count = fitness_evaluations_NB_count.values.tolist()
        fitness_evaluations_NB_count = sum(fitness_evaluations_NB_count) * 5
        evaluations_list_NB.append(fitness_evaluations_NB_count)

    # add information to the dicts
    hv_dict_WS.append(hv_list_WS)
    hv_dict_SR.append(hv_list_SR)
    hv_dict_RF.append(hv_list_RF)
    hv_dict_DT.append(hv_list_DT)
    hv_dict_NB.append(hv_list_NB)
    evaluations_dict_WS.append(evaluations_list_WS)
    evaluations_dict_SR.append(evaluations_list_SR)
    evaluations_dict_RF.append(evaluations_list_RF)
    evaluations_dict_DT.append(evaluations_list_DT)
    evaluations_dict_NB.append(evaluations_list_NB)

    # plot the convergence curve
    #plt.plot(evaluations_list_WS, hv_list_WS, label='WS')
    #plt.plot(evaluations_list_RF, hv_list_RF, label='RF')
    #plt.plot(evaluations_list_SR, hv_list_SR, label='SR')
    #plt.title(f'Run {j}')
    #plt.grid()
    #plt.legend()
    #plt.xlim(left=200)
    #plt.xlim(right=5000)
    #plt.savefig(f'Run_{j}.png')
    #plt.show()

# calculate the mean over all runs
hv_list_WS_average = [0]
hv_list_RF_average = [0]
hv_list_SR_average = [0]
hv_list_DT_average = [0]
hv_list_NB_average = [0]

hv_list_WS_average.extend([mean([item[j] for item in hv_dict_WS]) for j in range(0,50)])
hv_list_RF_average.extend([mean([item[j] for item in hv_dict_RF]) for j in range(0,140)])
hv_list_SR_average.extend([mean([item[j] for item in hv_dict_SR]) for j in range(0,50)])
hv_list_DT_average.extend([mean([item[j] for item in hv_dict_DT]) for j in range(0,140)])
hv_list_NB_average.extend([mean([item[j] for item in hv_dict_NB]) for j in range(0,140)])

evaluations_list_WS_average = [0]
evaluations_list_RF_average = [0]
evaluations_list_SR_average = [0]
evaluations_list_DT_average = [0]
evaluations_list_NB_average = [0]

evaluations_list_WS_average.extend([mean([item[j] for item in evaluations_dict_WS]) for j in range(0,50)])
evaluations_list_RF_average.extend([mean([item[j] for item in evaluations_dict_RF]) for j in range(0,140)])
evaluations_list_SR_average.extend([mean([item[j] for item in evaluations_dict_SR]) for j in range(0,50)])
evaluations_list_DT_average.extend([mean([item[j] for item in evaluations_dict_DT]) for j in range(0,140)])
evaluations_list_NB_average.extend([mean([item[j] for item in evaluations_dict_NB]) for j in range(0,140)])

# plot the convergence curve for the mean over all runs
plt.figure(figsize=(8, 5))
plt.plot(evaluations_list_WS_average, hv_list_WS_average, label= 'MO-GP-HH$\mathregular{_{WS}}$')
plt.plot(evaluations_list_SR_average, hv_list_SR_average, label= 'MO-GP-HH$\mathregular{_{SR}}$')
plt.plot(evaluations_list_RF_average, hv_list_RF_average, label= 'MO-GP-HH$\mathregular{_{RF}}$')
plt.plot(evaluations_list_DT_average, hv_list_DT_average, label= 'MO-GP-HH$\mathregular{_{DT}}$')
plt.plot(evaluations_list_NB_average, hv_list_NB_average, label= 'MO-GP-HH$\mathregular{_{NB}}$')
plt.grid()
plt.legend(fontsize=11, loc='lower right')
plt.xlabel('Simulation Replications', fontsize=11)
plt.ylabel('HVR', fontsize=11)
plt.xlim(left=0)
plt.xlim(right=25000)
plt.ylim(bottom=0.965, top=1.000)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
#plt.title('Mean convergence of 30 independent runs')
plt.tight_layout()
plt.savefig(f'Convergence_curve.png', dpi=600)
plt.show()

# plot boxplot of the mean of all runs
hv_list_WS_boxplot = [max(liste) for liste in hv_dict_WS]
hv_list_RF_boxplot = [max(liste) for liste in hv_dict_RF]
hv_list_SR_boxplot = [max(liste) for liste in hv_dict_SR]
hv_list_DT_boxplot = [max(liste) for liste in hv_dict_DT]
hv_list_NB_boxplot = [max(liste) for liste in hv_dict_NB]

data = [hv_list_WS_boxplot, hv_list_SR_boxplot, hv_list_RF_boxplot, hv_list_DT_boxplot, hv_list_NB_boxplot]
fig1, ax1 = plt.subplots(figsize=(7, 5))
bplot = ax1.boxplot(data, notch=True, patch_artist=True)
colors = ['lightblue', 'wheat', 'lightgreen', 'tomato', 'violet']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax1.grid(True)
ax1.set_ylabel('HVR', fontsize=11)
ax1.set_xticks([1, 2, 3, 4, 5], labels=['MO-GP-HH$\mathregular{_{WS}}$', 'MO-GP-HH$\mathregular{_{SR}}$', 'MO-GP-HH$\mathregular{_{RF}}$', 'MO-GP-HH$\mathregular{_{DT}}$', 'MO-GP-HH$\mathregular{_{NB}}$'], fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(bottom=0.970, top=1.000)
plt.tight_layout()
plt.savefig('Training_HV.png', dpi=600)
plt.show()

# perform wicoxon test
w, p = wilcoxon(hv_list_SR_boxplot, hv_list_WS_boxplot)
print(f'Wilcoxon difference SR to WS: {w}, {p}')
w, p = wilcoxon(hv_list_RF_boxplot, hv_list_WS_boxplot)
print(f'Wilcoxon difference RF to WS: {w}, {p}')
w, p = wilcoxon(hv_list_RF_boxplot, hv_list_SR_boxplot)
print(f'Wilcoxon difference RF to SR: {w}, {p}')

w, p = wilcoxon(hv_list_SR_boxplot, hv_list_WS_boxplot, alternative='greater')
print(f'Wilcoxon SR greater than WS: {w}, {p}')
w, p = wilcoxon(hv_list_RF_boxplot, hv_list_WS_boxplot, alternative='greater')
print(f'Wilcoxon RF greater than WS: {w}, {p}')
w, p = wilcoxon(hv_list_RF_boxplot, hv_list_SR_boxplot, alternative='greater')
print(f'Wilcoxon RF grater than SR: {w}, {p}')

# print results of mean and standard deviation of the achieved HVR
print('HVR')
print(f'WS mean: {mean(hv_list_WS_boxplot)} std: {std(hv_list_WS_boxplot)}')
print(f'SR mean: {mean(hv_list_SR_boxplot)} std: {std(hv_list_SR_boxplot)}')
print(f'RF mean: {mean(hv_list_RF_boxplot)} std: {std(hv_list_RF_boxplot)}')
print(f'DT mean: {mean(hv_list_DT_boxplot)} std: {std(hv_list_DT_boxplot)}')
print(f'NB mean: {mean(hv_list_NB_boxplot)} std: {std(hv_list_NB_boxplot)}')


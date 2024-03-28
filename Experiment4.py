import pygmo as pg
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from numpy import std

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
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/WS/run_{str(j)}/testing.xlsx'
    fitness_WS = pd.read_excel(link, header=0, index_col=0)
    # RF
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/RF_duplicate/run_{str(j)}/testing.xlsx'
    fitness_RF = pd.read_excel(link, header=0, index_col=0)
    #SR
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/SR/run_{str(j)}/testing.xlsx'
    fitness_SR = pd.read_excel(link, header=0, index_col=0)
    #DT
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/DT/run_{str(j)}/testing.xlsx'
    fitness_DT = pd.read_excel(link, header=0, index_col=0)
    #NB
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/NB/run_{str(j)}/testing.xlsx'
    fitness_NB = pd.read_excel(link, header=0, index_col=0)

    # drop na
    f1f2_WS = fitness_WS[['mean flowtime','max tardiness']]
    f1f2_WS = f1f2_WS.dropna()
    f1f2_SR = fitness_SR[['mean flowtime','max tardiness']]
    f1f2_SR = f1f2_SR.dropna()
    f1f2_RF = fitness_RF[['mean flowtime','max tardiness']]
    f1f2_RF = f1f2_RF.dropna()
    f1f2_DT = fitness_DT[['mean flowtime','max tardiness']]
    f1f2_DT = f1f2_DT.dropna()
    f1f2_NB = fitness_NB[['mean flowtime','max tardiness']]
    f1f2_NB = f1f2_NB.dropna()

    f1f2_WS = f1f2_WS.values.tolist()
    f1f2_SR = f1f2_SR.values.tolist()
    f1f2_RF = f1f2_RF.values.tolist()
    f1f2_DT = f1f2_DT.values.tolist()
    f1f2_NB = f1f2_NB.values.tolist()

    f1f2_reference.extend(f1f2_WS)
    f1f2_reference.extend(f1f2_SR)
    f1f2_reference.extend(f1f2_RF)
    f1f2_reference.extend(f1f2_DT)
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

hv_list_WS = []
hv_list_RF = []
hv_list_SR = []
hv_list_DT = []
hv_list_NB = []

for j in range(1,31):
    print(f'Run: {j}')

    # load files
    # WS
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/WS/run_{str(j)}/testing.xlsx'
    fitness_WS = pd.read_excel(link, header=0, index_col=0)
    # RF
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/RF_duplicate/run_{str(j)}/testing.xlsx'
    fitness_RF = pd.read_excel(link, header=0, index_col=0)
    #SR
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/SR/run_{str(j)}/testing.xlsx'
    fitness_SR = pd.read_excel(link, header=0, index_col=0)
    #DT
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/DT/run_{str(j)}/testing.xlsx'
    fitness_DT = pd.read_excel(link, header=0, index_col=0)
    #NB
    link = f'D:/PycharmProjects/08_EMO_hyper_heuristic/Results/NB/run_{str(j)}/testing.xlsx'
    fitness_NB = pd.read_excel(link, header=0, index_col=0)

    # drop na
    f1f2_WS = fitness_WS[['mean flowtime','max tardiness']]
    f1f2_WS = f1f2_WS.dropna()
    f1f2_SR = fitness_SR[['mean flowtime','max tardiness']]
    f1f2_SR = f1f2_SR.dropna()
    f1f2_RF = fitness_RF[['mean flowtime','max tardiness']]
    f1f2_RF = f1f2_RF.dropna()
    f1f2_DT = fitness_DT[['mean flowtime','max tardiness']]
    f1f2_DT = f1f2_DT.dropna()
    f1f2_NB = fitness_NB[['mean flowtime','max tardiness']]
    f1f2_NB = f1f2_NB.dropna()

    # calculate hypervolume
    # WS
    hyp_WS = pg.hypervolume(f1f2_WS.values)
    hv_WS = hyp_WS.compute([f1_nadir, f2_nadir])
    hv_WS = hv_WS/hv_ref
    hv_list_WS.append(hv_WS)
    # SR
    hyp_SR = pg.hypervolume(f1f2_SR.values)
    hv_SR = hyp_SR.compute([f1_nadir, f2_nadir])
    hv_SR = hv_SR/hv_ref
    hv_list_SR.append(hv_SR)
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


# plot boxplot of the mean of all runs
data = [hv_list_WS, hv_list_SR, hv_list_RF, hv_list_DT, hv_list_NB]
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
plt.savefig('Testing_HV.png', dpi=600)
plt.show()

# print results of mean and standard deviation of the achieved HVR
print('HVR')
print(f'WS mean: {mean(hv_list_WS)} std: {std(hv_list_WS)}')
print(f'SR mean: {mean(hv_list_SR)} std: {std(hv_list_SR)}')
print(f'RF mean: {mean(hv_list_RF)} std: {std(hv_list_RF)}')
print(f'DT mean: {mean(hv_list_DT)} std: {std(hv_list_DT)}')
print(f'NB mean: {mean(hv_list_NB)} std: {std(hv_list_NB)}')
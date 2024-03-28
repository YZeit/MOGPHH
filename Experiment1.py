import pandas as pd
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import std, mean

selection_accuracy = []
KNN = []
MLP = []
DT = []
SVM = []
NB = []
LR = []
RF = []
halfshop = []
singlerep = []
singlerepshort = []
for i in range(30):
    link = f'C:/Users/yanni/OneDrive/PhD Engineering and Management/surrogate-assisted MO-GP-HH/Experiments/Experiment1 - selection accuracy and execution time/run_{i}/selection_accuracy.xlsx'
    selection_accuracy = pd.read_excel(link, header=0, index_col=0)
    #print(selection_accuracy)
    #print(list(selection_accuracy['KNN']))
    KNN.append(mean(list(selection_accuracy['KNN'])))
    MLP.append(mean(list(selection_accuracy['MLP'])))
    DT.append(mean(list(selection_accuracy['DT'])))
    SVM.append(mean(list(selection_accuracy['SVM'])))
    NB.append(mean(list(selection_accuracy['NB'])))
    LR.append(mean(list(selection_accuracy['LR'])))
    RF.append(mean(list(selection_accuracy['RF'])))
    halfshop.append(mean(list(selection_accuracy['Halfshop'])))
    singlerep.append(mean(list(selection_accuracy['Singlerep'])))
    singlerepshort.append(mean(list(selection_accuracy['Singlerepshort'])))

# fake up some data
data = [singlerep, singlerepshort, halfshop, KNN, MLP, DT, SVM, NB, LR, RF]

print(data)

fig, ax1 = plt.subplots(figsize =(8, 5))
# Creating axes instance
#ax = fig.add_axes([0, 0, 1, 1])
# Creating plot
#plt.boxplot(data, notch=True)
bplot = ax1.boxplot(data, notch=True, patch_artist=True)
colors = ['wheat', 'wheat', 'wheat', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.ylim(top=1)
plt.grid(True)
plt.ylabel('Selection Accuracy', fontsize=11)
#plt.title('Selection accuracy')
#plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['SR', 'SR_short', 'HS', 'KNN', 'MLP', 'DT', 'SVM', 'NB', 'LR', 'RF'], fontsize=11)
plt.tight_layout()
plt.savefig('Selection_accuracy.png', dpi=600)
plt.show()


from matplotlib.ticker import MultipleLocator
import operator
import matplotlib.ticker as mtick

link = f'C:/Users/yanni/OneDrive/PhD Engineering and Management/surrogate-assisted MO-GP-HH/Experiments/Experiment1 - selection accuracy and execution time/time.xlsx'
time = pd.read_excel(link, header=0, index_col=0)
print(time)
phenotypic_time_1 = list(time['phenotypic 1'])
phenotypic_time_2 = list(time['phenotypic 2'])
KNN_time_2 = list(time['KNN train'])
MLP_time_2 = list(time['MLP train'])
DT_time_2 = list(time['DT train'])
SVM_time_2 = list(time['SVM train'])
NB_time_2 = list(time['NB train'])
LR_time_2 = list(time['LR train'])
RF_time_2 = list(time['RF train'])
KNN_time_1 = list(time['KNN predict'])
MLP_time_1 = list(time['MLP predict'])
DT_time_1 = list(time['DT predict'])
SVM_time_1 = list(time['SVM predict'])
NB_time_1 = list(time['NB predict'])
LR_time_1 = list(time['LR predict'])
RF_time_1 = list(time['RF predict'])
SR_time = list(time['SR'])
SR_short_time = list(time['SR_short'])
HS_time = list(time['HS'])
full_evaluation = list(time['Full Evaluation'])

phenotypic_time = list(map(operator.add, phenotypic_time_1, phenotypic_time_2))

KNN_time = list(map(operator.add, KNN_time_1, KNN_time_2))
MLP_time = list(map(operator.add, MLP_time_1, MLP_time_2))
DT_time = list(map(operator.add, DT_time_1, DT_time_2))
SVM_time = list(map(operator.add, SVM_time_1, SVM_time_2))
NB_time = list(map(operator.add, NB_time_1, NB_time_2))
LR_time = list(map(operator.add, LR_time_1, LR_time_2))
RF_time = list(map(operator.add, RF_time_1, RF_time_2))

KNN_time = list(map(operator.add, KNN_time, phenotypic_time))
MLP_time = list(map(operator.add, MLP_time, phenotypic_time))
DT_time = list(map(operator.add, DT_time, phenotypic_time))
SVM_time = list(map(operator.add, SVM_time, phenotypic_time))
NB_time = list(map(operator.add, NB_time, phenotypic_time))
LR_time = list(map(operator.add, LR_time, phenotypic_time))
RF_time = list(map(operator.add, RF_time, phenotypic_time))


KNN_time = list(map(operator.truediv, KNN_time, full_evaluation))
MLP_time = list(map(operator.truediv, MLP_time, full_evaluation))
DT_time = list(map(operator.truediv, DT_time, full_evaluation))
SVM_time = list(map(operator.truediv, SVM_time, full_evaluation))
NB_time = list(map(operator.truediv, NB_time, full_evaluation))
LR_time = list(map(operator.truediv, LR_time, full_evaluation))
RF_time = list(map(operator.truediv, RF_time, full_evaluation))

SR_time = list(map(operator.truediv, SR_time, full_evaluation))
SR_short_time = list(map(operator.truediv, SR_short_time, full_evaluation))
HS_time = list(map(operator.truediv, HS_time, full_evaluation))

data = [SR_time, SR_short_time, HS_time, KNN_time, MLP_time, DT_time, SVM_time, NB_time, LR_time, RF_time]

fig, ax1 = plt.subplots(figsize =(8, 5))
bplot = ax1.boxplot(data, notch=True, patch_artist=True)
colors = ['wheat', 'wheat', 'wheat', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_yticks(np.arange(0, 0.25, step=0.05))
ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['SR', 'SR_short', 'HS', 'KNN', 'MLP', 'DT', 'SVM', 'NB', 'LR', 'RF'], fontsize=11)
ax1.yaxis.get_ticklocs(minor=True)
ax1.minorticks_on()
ax1.xaxis.set_tick_params(which='minor', bottom=False)
#plt.legend(fontsize=11)
plt.ylabel('Execution Time Ratio', fontsize=11)
ax1.grid(True)
ax1.set_yscale('log')
ax1.set_yticks([0.01, 0.02, 0.05, 0.10, 0.20])
vals = ax1.get_yticks()
ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals], fontsize=11)
ax1.tick_params(axis='y', which='minor', bottom=False)
ax1.grid(True, axis='y', which='both')
plt.tight_layout()
plt.savefig('Selection_accuracy_2.png', dpi=600)
plt.show()


# print mean and standard deviation of all results
print('Selection Accuracy')
print(f'SR mean: {mean(singlerep)} std: {std(singlerep)}')
print(f'SR_short mean: {mean(singlerepshort)} std: {std(singlerepshort)}')
print(f'HS mean: {mean(halfshop)} std: {std(halfshop)}')
print(f'KNN mean: {mean(KNN)} std: {std(KNN)}')
print(f'MLP mean: {mean(MLP)} std: {std(MLP)}')
print(f'DT mean: {mean(DT)} std: {std(DT)}')
print(f'SVM mean: {mean(SVM)} std: {std(SVM)}')
print(f'NB mean: {mean(NB)} std: {std(NB)}')
print(f'LR mean: {mean(LR)} std: {std(LR)}')
print(f'RF mean: {mean(RF)} std: {std(RF)}')

print('Execution Time')
print(f'SR mean: {mean(SR_time)} std: {std(SR_time)}')
print(f'SR_short mean: {mean(SR_short_time)} std: {std(SR_short_time)}')
print(f'HS mean: {mean(HS_time)} std: {std(HS_time)}')
print(f'KNN mean: {mean(KNN_time)} std: {std(KNN_time)}')
print(f'MLP mean: {mean(MLP_time)} std: {std(MLP_time)}')
print(f'DT mean: {mean(DT_time)} std: {std(DT_time)}')
print(f'SVM mean: {mean(SVM_time)} std: {std(SVM_time)}')
print(f'NB mean: {mean(NB_time)} std: {std(NB_time)}')
print(f'LR mean: {mean(LR_time)} std: {std(LR_time)}')
print(f'RF mean: {mean(RF_time)} std: {std(RF_time)}')


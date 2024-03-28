import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import random
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# function to train the Logistic Regression Classifier
def LR_train(X, y):
    y = np.array(y, dtype='str')
    y = [",".join(item) for item in y.astype(str)]
    LR_model = LogisticRegression()
    LR_model.fit(X,y)
    return LR_model

# function to train the Random Forest Classifier
def RF_train(X, y):
    y = np.array(y, dtype='str')
    y = [",".join(item) for item in y.astype(str)]
    RF_model = RandomForestClassifier()
    RF_model.fit(X,y)
    return RF_model

# function to train the Naive Bayes Classifier
def NB_train(X, y):
    y = np.array(y, dtype='str')
    y = [",".join(item) for item in y.astype(str)]
    NB_model = GaussianNB()
    NB_model.fit(X,y)
    return NB_model

# function to train the Decision Tree Classifier
def DT_train(X, y):
    y = np.array(y, dtype='str')
    y = [",".join(item) for item in y.astype(str)]
    DT_model = DecisionTreeClassifier()
    DT_model.fit(X,y)
    return DT_model

# function to train the K-Nearest-Neighbor Classifier
def KNN_train(X, y):
    y = np.array(y, dtype='str')
    y = [",".join(item) for item in y.astype(str)]
    KNN_model = KNeighborsClassifier(n_neighbors=1, p=2)
    KNN_model.fit(X,y)
    return KNN_model

# function to train the Support Vector Machine classifier
def SVM_train(X, y):
    y = np.array(y, dtype='str')
    y = [",".join(item) for item in y.astype(str)]
    SVM_model = svm.LinearSVC()
    SVM_model.fit(X,y)
    return SVM_model

# function to train the Multi-Layer_perception Classifier (simple Artificial Neural Network)
def MLP_train(X, y):
    y = np.array(y, dtype='str')
    y = [",".join(item) for item in y.astype(str)]
    #MLP_model = MLPClassifier(hidden_layer_sizes=(150,100,50),
    #                    max_iter = 300,activation = 'relu',
    #                    solver = 'adam')
    MLP_model = MLPClassifier()
    MLP_model.fit(X,y)
    return MLP_model

# function to predict a particular dataset on an trained input model
def predict(model, data):
    prediction = model.predict(data)
    prediction_formatted = []
    for i in range(len(prediction)):
        x = prediction[i].split(",")
        x[0] = float(x[0])
        x[1] = float(x[1])
        prediction_formatted.append((x[0], x[1]))
    return prediction_formatted

# function to evaluate the accuracy of a surrogate models
def evaluate_selection_accuracy(ranking_actual, ranking_predicted, pop_size_offspring, pop_size_parents):
    accuracy = 0
    correct_predictions = 0
    expected_correct_predictions = (pop_size_parents*pop_size_parents)/pop_size_offspring
    #expected_accuracy = (pop_size_parents*pop_size_offspring)/2
    expected_accuracy = ((pop_size_parents*pop_size_parents)/pop_size_offspring) * (((pop_size_offspring-1)+(pop_size_offspring-pop_size_parents))/2)
    max_accuracy = pop_size_parents*(pop_size_offspring-((pop_size_parents+1)/2))
    for i in ranking_actual:
        if i in ranking_predicted:
            accuracy += (pop_size_offspring-(ranking_actual.index(i)+1))
            correct_predictions += 1
    selection_accuracy = (accuracy-expected_accuracy)/(max_accuracy-expected_accuracy)
    selection_accuracy_simple = (correct_predictions-expected_correct_predictions)/(pop_size_parents-expected_correct_predictions)
    return selection_accuracy, selection_accuracy_simple

def evaluate_rank_correlation(ranking_actual, ranking_predicted, pop_size_offspring):
    sum_of_squared_difference = 0
    for i in ranking_predicted:
        rank_predicted = ranking_predicted.index(i)
        rank_actual = ranking_actual.index(i)
        difference = rank_predicted-rank_actual
        difference_square = difference*difference
        sum_of_squared_difference += difference_square
    rank_correlation = 1 - ((12*sum_of_squared_difference) / ((pop_size_offspring) * ((pop_size_offspring*pop_size_offspring)-1)))
    return rank_correlation

'''
# test the predictors
decision_vectors = pd.read_csv("decision_vectors.csv", index_col=0)
decision_vectors = np.array(decision_vectors)
trainingsdata = decision_vectors[:76, :]
testdata = decision_vectors[77:, :-2]

# train the KNN
KNN_model = KNN_train(trainingsdata)
KNN_prediction = predict(KNN_model, data=testdata)
print('Prediction KNN')
print(KNN_prediction)
print('\n')

# train the DT
DT_model = DT_train(trainingsdata)
DT_prediction = predict(DT_model, data=testdata)
print('Prediction DT')
print(DT_prediction)
print('\n')

# train the MLP
MLP_model = MLP_train(trainingsdata)
MLP_prediction = predict(MLP_model, data=testdata)
print('Prediction MLP')
print(MLP_prediction)
print('\n')

# train the SVM
SVM_model = SVM_train(trainingsdata)
SVM_prediction = predict(SVM_model, data=testdata)
print('Prediction SVM')
print(SVM_prediction)
print('\n')

# test evaluation
original_population = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ranking_actual = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
selection_accuracy = []
selection_accuracy_simple = []
rank_correlation = []
ranking_predicted = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
#ranking_predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#ranking_predicted = [20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
for i in range(100000):
    #ranking_predicted = random.sample(original_population, 20)
    #ranking_predicted = ranking_predicted[:10]
    selection_accuracy_c, selection_accuracy_simple_c = evaluate_selection_accuracy(ranking_actual=ranking_actual, ranking_predicted=ranking_predicted, pop_size_offspring=len(original_population), pop_size_parents=len(ranking_actual))
    selection_accuracy.append(selection_accuracy_c)
    selection_accuracy_simple.append(selection_accuracy_simple_c)
    rank_correlation_c = evaluate_rank_correlation(ranking_actual=original_population, ranking_predicted=ranking_predicted, pop_size_offspring=len(original_population))
x = np.mean(selection_accuracy)
y = np.mean(selection_accuracy_simple)
z = np.mean(rank_correlation_c)
print(f'selection accuracy: {x}')
print(f'selection accuracy simple: {y}')
print(f'rank correlation: {z}')

'''


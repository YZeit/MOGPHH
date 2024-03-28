import scipy.stats as ss
from simulation import simulation
import random
import pandas as pd


# generates a random set of decision from a real simulation scenario
def decision_situations_generator(number_of_situations=100):
    decision_situations_dict = simulation(number_machines=10, number_jobs=1000, warm_up=100, func=0, due_date_tightness=4,
                                          utilization=0.80, decision_situtation=True, missing_operation=True)
    decision_situations_df = pd.DataFrame(decision_situations_dict)
    max_decision_situations = max(decision_situations_dict['decision'])
    frames = []
    random_value = random.sample(range(1, max_decision_situations), 1000)
    #frames = [decision_situations_df[decision_situations_df['decision']==random_value[i]] for i in random_value]
    # select random decisions from all decision situations
    for i in range(number_of_situations):
        df_temp = decision_situations_df[decision_situations_df['decision']==random_value[i]]
        frames.append(df_temp)
    #print(len(frames))
    #frames = list(dict.fromkeys(frames))
    #print(len(frames))
    decision_situations = pd.concat(frames)
    return decision_situations

# generates a ranking vector of an individual on a set of decisions
def ranking_vector_generator(individual_func, decision_situations):
    ranking_vector = []
    decision_situations_list = list(decision_situations['decision'])
    # remove duplicates from list
    decision_situations_list = list(dict.fromkeys(decision_situations_list))
    #print(decision_situations_list)
    #print(len(decision_situations_list))
    for i in decision_situations_list:
        decision_dict = decision_situations[decision_situations['decision'] == i]
        # rankings_vector_ref = list(decision_dict['reference rank'])
        # print(decision_dict_temp)
        priority_vector = []
        for index, row in decision_dict.iterrows():
            PT = row['PT']
            RT = row['RT']
            RPT = row['RPT']
            RNO = row['RNO']
            DD = row['DD']
            RTO = row['RTO']
            PTN = row['PTN']
            SL = row['SL']
            WT = row['WT']
            APTQ = row['APTQ']
            NJQ = row['NJQ']
            WINQ = row['WINQ']
            CT = row['CT']
            priority = individual_func(PT, RT, RPT, RNO, DD, RTO, PTN, SL, WT, APTQ, NJQ, WINQ, CT)
            #priority = PT
            # Here the reference priority function needs to be defined
            priority_vector.append(priority)
        # print(priority_vector)
        rankings = ss.rankdata(priority_vector, method='ordinal')
        ranking_vector.extend(rankings)
        #print(priority_vector)
        #print(rankings)
        #print(ranking_vector)
        # print(rankings_vector_ref)
        # print(rankings)
        # print(decision_vector_element)
    return ranking_vector

# dummy function for testing
def ranking_vector_generator_ref(individual_func, decision_situations):
    ranking_vector = []
    decision_situations_list = list(decision_situations['decision'])
    # remove duplicates from list
    decision_situations_list = list(dict.fromkeys(decision_situations_list))
    #print(decision_situations_list)
    #print(len(decision_situations_list))
    for i in decision_situations_list:
        decision_dict = decision_situations[decision_situations['decision'] == i]
        # rankings_vector_ref = list(decision_dict['reference rank'])
        # print(decision_dict_temp)
        priority_vector = []
        for index, row in decision_dict.iterrows():
            PT = row['PT']
            RT = row['RT']
            RPT = row['RPT']
            RNO = row['RNO']
            DD = row['DD']
            RTO = row['RTO']
            PTN = row['PTN']
            SL = row['SL']
            WT = row['WT']
            APTQ = row['APTQ']
            NJQ = row['NJQ']
            WINQ = row['WINQ']
            CT = row['CT']
            # priority = individual_func(PT, RPT, RNO, DD, SPTQ, APTQ, MAXPTQ, MINPTQ, MAXDDQ, NJQ, SPT, TRNO, CT)
            priority = PT
            # Here the reference priority function needs to be defined
            priority_vector.append(priority)
        # print(priority_vector)
        rankings = ss.rankdata(priority_vector, method='ordinal')
        ranking_vector.extend(rankings)
        #print(priority_vector)
        #print(rankings)
        #print(ranking_vector)
        # print(rankings_vector_ref)
        # print(rankings)
        # print(decision_vector_element)
    return ranking_vector

# generates the decision vector from two ranking vectors (reference and the individual's under consideration)
def decision_vector_generator(ranking_vector_ref, ranking_vector):
    decision_vector = []
    for i in range(len(ranking_vector)):
        ranking = ranking_vector[i]
        if ranking==1:
            decision_variable = ranking_vector_ref[i]
            decision_vector.append(decision_variable)
    return decision_vector





'''
#testing the functions
decision_situations = decision_situations_generator()
print(decision_situations)
ranking_vector = ranking_vector_generator(individual_func=0, decision_situations=decision_situations)
ranking_vector_ref = ranking_vector_generator_ref(individual_func=0, decision_situations=decision_situations)
print(ranking_vector)
print(len(ranking_vector))
print(ranking_vector_ref)
print(len(ranking_vector_ref))
decision_vector = decision_vector_generator(ranking_vector_ref=ranking_vector_ref, ranking_vector=ranking_vector)
print(decision_vector)
print(len(decision_vector))
'''

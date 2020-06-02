import utils
import numpy as np
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: accuracy
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}
    npp = []
    npp_data = {}

    for threshold in np.arange(0,1,0.01):
        for key,value in categorical_results.items():
            t_data = utils.apply_threshold(value,threshold)
            npp = (utils.get_num_predicted_positives(t_data)/len(t_data))
            if(key not in npp_data):
                npp_data[key] = []
            npp_data[key].append([npp,threshold])
    
    keys = [*npp_data]
    npp_data_refined = []
    for npp_d_0 in npp_data[keys[0]]:
        for npp_d_1 in npp_data[keys[1]]:
            if(abs(npp_d_0[0]-npp_d_1[0]) <= epsilon):
                npp_data_refined.append([npp_d_0,npp_d_1])

    npp_data_refined_2 = []
    for val in npp_data_refined:
        for npp_d_2 in npp_data[keys[2]]:
            if(abs(npp_d_2[0] - val[0][0]) <= epsilon):
                if(abs(npp_d_2[0] - val[1][0]) <= epsilon):
                    npp_data_refined_2.append([val[0],val[1],npp_d_2])

    npp_data_refined_3 = []
    for val in npp_data_refined_2:
        for npp_d_3 in npp_data[keys[3]]:
            if(abs(npp_d_3[0] - val[0][0]) <= epsilon):
                if(abs(npp_d_3[0] - val[1][0]) <= epsilon):
                    if(abs(npp_d_3[0] - val[2][0]) <= epsilon):
                        npp_data_refined_3.append([val[0],val[1],val[2],npp_d_3])
    
    max_acc = 0
    temp = {}
    for thresh in npp_data_refined_3:
        temp['African-American'] = utils.apply_threshold(categorical_results['African-American'],thresh[0][1])
        temp['Caucasian'] = utils.apply_threshold(categorical_results['Caucasian'],thresh[1][1])
        temp['Hispanic'] = utils.apply_threshold(categorical_results['Hispanic'],thresh[2][1])
        temp['Other'] = utils.apply_threshold(categorical_results['Other'],thresh[3][1])
        acc = utils.get_total_accuracy(temp)
        if(acc > max_acc):
            max_acc = acc
            thresholds['African-American'] = thresh[0][1]
            thresholds['Caucasian'] = thresh[1][1]
            thresholds['Hispanic'] = thresh[2][1]
            thresholds['Other'] = thresh[3][1]
            demographic_parity_data = temp.copy()

    max_acc = 0
    return demographic_parity_data, thresholds

    
    # Must complete this function!
    #return demographic_parity_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):
    thresholds = {}
    equal_opportunity_data = {}

    tpr_data = {}
    
    for threshold in np.arange(0,1,0.01):
        for key,value in categorical_results.items():
            t_data = utils.apply_threshold(value,threshold)
            tpr = utils.get_true_positive_rate(t_data)
            if(key not in tpr_data):
                tpr_data[key] = []
            tpr_data[key].append([tpr,threshold])
    
    keys = [*tpr_data]
    tpr_data_refined = []
    for tpr_d_0 in tpr_data[keys[0]]:
        for tpr_d_1 in tpr_data[keys[1]]:
            if(abs(tpr_d_0[0]-tpr_d_1[0]) <= epsilon):
                tpr_data_refined.append([tpr_d_0,tpr_d_1])

    tpr_data_refined_2 = []
    for val in tpr_data_refined:
        for tpr_d_2 in tpr_data[keys[2]]:
            if(abs(tpr_d_2[0] - val[0][0]) <= epsilon):
                if(abs(tpr_d_2[0] - val[1][0]) <= epsilon):
                    tpr_data_refined_2.append([val[0],val[1],tpr_d_2])

    tpr_data_refined_3 = []
    for val in tpr_data_refined_2:
        for tpr_d_3 in tpr_data[keys[3]]:
            if(abs(tpr_d_3[0] - val[0][0]) <= epsilon):
                if(abs(tpr_d_3[0] - val[1][0]) <= epsilon):
                    if(abs(tpr_d_3[0] - val[2][0]) <= epsilon):
                        tpr_data_refined_3.append([val[0],val[1],val[2],tpr_d_3])
    
    #print(len(tpr_data_refined_3))
    max_acc = 0
    temp = {}
    for thresh in tpr_data_refined_3:
        temp['African-American'] = utils.apply_threshold(categorical_results['African-American'],thresh[0][1])
        temp['Caucasian'] = utils.apply_threshold(categorical_results['Caucasian'],thresh[1][1])
        temp['Hispanic'] = utils.apply_threshold(categorical_results['Hispanic'],thresh[2][1])
        temp['Other'] = utils.apply_threshold(categorical_results['Other'],thresh[3][1])
        acc = utils.get_total_accuracy(temp)
        if(acc > max_acc):
            max_acc = acc
            thresholds['African-American'] = thresh[0][1]
            thresholds['Caucasian'] = thresh[1][1]
            thresholds['Hispanic'] = thresh[2][1]
            thresholds['Other'] = thresh[3][1]
            equal_opportunity_data = temp.copy()

    return equal_opportunity_data, thresholds


    # Must complete this function!
    #return equal_opportunity_data, thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}
    
    for key, value in categorical_results.items():
        li = []
        max_acc = 0
        threshold = 0
        max_range_thresh = max(value, key=lambda x:x[0])
        min_range_thresh = min(value, key=lambda x:x[0])
        for thresh in np.arange(min_range_thresh[0], max_range_thresh[0], 0.01):
            result = utils.apply_threshold(value,thresh)
            total_num_cases = 0
            total_correct = 0
            for prediction, label in result:
                    total_num_cases += 1.0
                    if prediction == label:
                        total_correct += 1.0
            acc = total_correct / total_num_cases
            # print(key, thresh, acc)
            if acc > max_acc:
                max_acc = acc
                threshold = thresh
        thresholds[key] = threshold
    for key, value in categorical_results.items():
        result = utils.apply_threshold(value,thresholds[key])
        mp_data[key] = result

    acc = utils.get_total_accuracy(mp_data)
    # print(acc)
    # Must complete this function!
    # return mp_data, thresholds

    return mp_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}
    thresholds_new = {}
    ppv = []
    max_range_thresh = [1,1,1,1]
    min_range_thresh = [0,0,0,0]
    threshold_set = []

    for key,val in categorical_results.items():
        max_range_thresh.append(max(categorical_results[key], key=lambda x:x[0])[0])
        min_range_thresh.append(min(categorical_results[key], key=lambda x:x[0])[0])

    ppv_data = {}

    for threshold in np.arange(0,1,0.01):
        for key,value in categorical_results.items():
            t_data = utils.apply_threshold(value,threshold)
            ppv = utils.get_positive_predictive_value(t_data)
            if(key not in ppv_data):
                ppv_data[key] = []
            ppv_data[key].append([ppv,threshold])
    
    keys = [*ppv_data]
    ppv_data_refined = []
    for ppv_d_0 in ppv_data[keys[0]]:
        for ppv_d_1 in ppv_data[keys[1]]:
            if(abs(ppv_d_0[0]-ppv_d_1[0]) <= epsilon):
                ppv_data_refined.append([ppv_d_0,ppv_d_1])

    ppv_data_refined_2 = []
    for val in ppv_data_refined:
        for ppv_d_2 in ppv_data[keys[2]]:
            if(abs(ppv_d_2[0] - val[0][0]) <= epsilon):
                if(abs(ppv_d_2[0] - val[1][0]) <= epsilon):
                    ppv_data_refined_2.append([val[0],val[1],ppv_d_2])

    ppv_data_refined_3 = []
    for val in ppv_data_refined_2:
        for ppv_d_3 in ppv_data[keys[3]]:
            if(abs(ppv_d_3[0] - val[0][0]) <= epsilon):
                if(abs(ppv_d_3[0] - val[1][0]) <= epsilon):
                    if(abs(ppv_d_3[0] - val[2][0]) <= epsilon):
                        ppv_data_refined_3.append([val[0],val[1],val[2],ppv_d_3])
    
    #print(len(ppv_data_refined_3))
    max_acc = 0
    temp = {}
    for thresh in ppv_data_refined_3:
        temp['African-American'] = utils.apply_threshold(categorical_results['African-American'],thresh[0][1])
        temp['Caucasian'] = utils.apply_threshold(categorical_results['Caucasian'],thresh[1][1])
        temp['Hispanic'] = utils.apply_threshold(categorical_results['Hispanic'],thresh[2][1])
        temp['Other'] = utils.apply_threshold(categorical_results['Other'],thresh[3][1])
        acc = utils.get_total_accuracy(temp)
        if(acc > max_acc):
            max_acc = acc
            thresholds['African-American'] = thresh[0][1]
            thresholds['Caucasian'] = thresh[1][1]
            thresholds['Hispanic'] = thresh[2][1]
            thresholds['Other'] = thresh[3][1]
            predictive_parity_data = temp.copy()


    max_acc = 0
    return predictive_parity_data, thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}
    single_thresh = 0
    merged = []
    max_acc = 0
    for k,v in categorical_results.items():
        merged.extend(v) 
    
    max_range_thresh = max(merged, key=lambda x:x[0])
    min_range_thresh = min(merged, key=lambda x:x[0])
        
    for thresh in np.arange(min_range_thresh[0] ,max_range_thresh[0] ,0.01):
        for key, value in categorical_results.items():
            result = utils.apply_threshold(value,thresh)
            single_threshold_data[key] = result
        acc = utils.get_total_accuracy(single_threshold_data)
        if(acc > max_acc):
            max_acc = acc
            single_thresh = thresh

    for key, value in categorical_results.items():
        thresholds[key] = single_thresh
        result = utils.apply_threshold(value,thresholds[key])
        single_threshold_data[key] = result
    # Must complete this function!
    #return single_threshold_data, thresholds

    return single_threshold_data, thresholds
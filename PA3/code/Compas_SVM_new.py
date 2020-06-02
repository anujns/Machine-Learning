from sklearn import svm
from Preprocessing import preprocess
from Postprocessing import *
from utils import *

metrics = ["sex", "age_cat", 'race', 'c_charge_degree', 'priors_count']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics, recalculate=False, causal=False)

np.random.seed(42)
SVR = svm.LinearSVR(C=1/float(len(test_data)), max_iter=5000)
SVR.fit(training_data, training_labels)

training_predictions = SVR.predict(training_data)
test_predictions = SVR.predict(test_data)

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases, 0.01)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

# ADD MORE PRINT LINES HERE - THIS ALONE ISN'T ENOUGH
# YOU NEED ACCURACY AND COST FOR TRAINING AND TEST DATA
# PLUS WHATEVER RELEVANT METRICS ARE USED IN YOUR POSTPROCESSING METHOD, TO ENSURE EPSILON WAS ENFORCED

for group in test_race_cases.keys():
    TPR = get_true_positive_rate(test_race_cases[group])
    print("TPR for " + group + ": " + str(TPR))

# for group in test_race_cases.keys():
#     num_positive_predictions = get_num_predicted_positives(test_race_cases[group])
#     prob = num_positive_predictions / len(test_race_cases[group])
#     print("Probability of positive prediction for " + str(group) + ": " + str(prob))

print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

print("Accuracy on test data:")
print(get_total_accuracy(test_race_cases))
print("")

print("Cost on test data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")

total_cost = apply_financials(training_race_cases) + apply_financials(test_race_cases)
print("Total Acc: ", total_cost)

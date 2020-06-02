from sklearn.naive_bayes import MultinomialNB
from Preprocessing import preprocess
from Postprocessing import *
from utils import *

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

NBC = MultinomialNB()
NBC.fit(training_data, training_labels)

training_class_predictions = NBC.predict_proba(training_data)
training_predictions = []
test_class_predictions = NBC.predict_proba(test_data)
test_predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i][1])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i][1])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases, 0.01)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

# ADD MORE PRINT LINES HERE - THIS ALONE ISN'T ENOUGH
# YOU NEED ACCURACY AND COST FOR TRAINING AND TEST DATA
# PLUS WHATEVER RELEVANT METRICS ARE USED IN YOUR POSTPROCESSING METHOD, TO ENSURE EPSILON WAS ENFORCED
print("Training TPRs: ")

for group in training_race_cases.keys():
    TPR1 = get_true_positive_rate(training_race_cases[group])
    print("TPR for " + group + ": " + str(TPR1))

print("Test TPRs: ")

for group in test_race_cases.keys():
    TPR2 = get_true_positive_rate(test_race_cases[group])
    print("TPR for " + group + ": " + str(TPR2))

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

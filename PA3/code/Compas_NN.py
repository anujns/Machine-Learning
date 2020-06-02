from numpy.random import seed
seed(4940)
from tensorflow import set_random_seed
set_random_seed(80)

import numpy as np
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from Preprocessing import preprocess
from Report_Results import report_results
from utils import *


def neural_network_classification(metrics):

    training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

    activation = "relu"
    model = Sequential()
    model.add(Dense(len(metrics)*2, activation=activation, kernel_regularizer=regularizers.l2(0.1), input_shape = (len(metrics), )))
    model.add(Dense(30, activation=activation, kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss="binary_crossentropy")
    model.fit(training_data, training_labels, epochs=30, batch_size=300, validation_data=(test_data, test_labels), verbose=0)

    data = np.concatenate((training_data, test_data))
    labels = np.concatenate((training_labels, test_labels))

    predictions = model.predict(data)
    predictions = np.squeeze(predictions, axis=1)

    return data, predictions, labels, categories, mappings

#######################################################################################################################


#######################################################################################################################


metrics = ["sex", "age_cat", "race", 'c_charge_degree', 'priors_count']

# Changing the int value sets the number of models to create before choosing the "best" one
data, predictions, labels, categories, mappings = neural_network_classification(metrics)
race_cases = get_cases_by_metric(data, categories, "race", mappings, predictions, labels)

report_results(race_cases)

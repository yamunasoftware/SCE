### INTERFACE IMPORTS ###

import numpy as np
import warnings
import os

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from ml import preprocessing
from ml import logistic

### INTERFACE SETUP ###

# Warnings Setup and Default Filename:
warnings.filterwarnings('ignore')
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
filename = os.path.join(parent_dir, "weights.npy")

### INTERFACE FUNCTIONS ###

# Default Lookup Table:
lookup_table = {
  -1: 'Negative',
  1: 'Positive'
}

# Classification Steps:
def classify(sentences):
  # Load the Weights:
  weights, message = train()
  feature_matrix = preprocessing.clean_set(sentences)

  # Predict labels for the sentences using the trained model
  predicted_labels = logistic.predict(feature_matrix, weights)
  return predicted_labels, message

# Classification Labels Output:
def label_output(predictions, lookup):
  new_predictions = []
  for prediction in predictions:
    if lookup is None:
      if prediction in lookup_table:
        new_predictions.append(lookup_table.get(prediction))
    else:
      if prediction in lookup:
        new_predictions.append(lookup.get(prediction))
  return new_predictions

# Main Training Function:
def train():
  data, labels, train_data, train_labels, test_data, test_labels = preprocessing.data_split()
  if find_weights(filename) is None:
    initial_weights = [0.0 for i in range(4501)]
    step_size = 0.1
    tolerance = 0.1

    final_weights = logistic.gradient_descent(train_data,train_labels, initial_weights, step_size, tolerance)
    np.save(filename, final_weights)

    train_predictions = logistic.predict(train_data, final_weights)
    test_predictions = logistic.predict(test_data, final_weights)
    message = '200 - ' + str(round(validate(train_predictions, train_labels), 2)) + ' ' + str(round(validate(test_predictions, test_labels), 2))

    return final_weights, message
  else:
    return find_weights(filename), '200 - Found Weights'
  
# Validation Function:
def validate(predictions, labels):
  count = 0
  i = 0

  while i < len(predictions):
    if predictions[i] == labels[i]:
      count += 1
    i += 1
  return (count / len(predictions)) * 100.0

# Finds the Weights:
def find_weights(file):
  if os.path.exists(file):
    return np.load(file)
  else:
    return None
  
# Gets Test Data Type:
def get_data_type(list, index):
  local_list = []
  for item in list:
    local_list.append(item[index])
  return local_list
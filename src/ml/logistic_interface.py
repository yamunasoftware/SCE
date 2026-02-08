### INTERFACE IMPORTS ###

import numpy as np
import os
import ml.preprocessing as preprocessing
import ml.logistic as logistic

### INTERFACE FUNCTIONS ###

# Default Weights Filename and Lookup Table:
filename = "/main/models/weights.npy"
lookup_table = {
  -1: 'Negative',
  1: 'Positive'
}

# Classification Steps:
def classify(sentences):
  # Predict labels for the sentences using the trained model (weights):
  weights = train()
  feature_matrix = preprocessing.clean_set(sentences)
  predicted_labels = logistic.predict(feature_matrix, weights)
  return predicted_labels

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
  _, _, train_data, train_labels, test_data, test_labels = preprocessing.data_split()
  if find_weights(filename) is None:
    initial_weights = [0.0 for i in range(4501)]
    step_size = 0.1
    tolerance = 0.1

    final_weights = logistic.gradient_descent(train_data,train_labels, initial_weights, step_size, tolerance)
    np.save(filename, final_weights)
    train_predictions = logistic.predict(train_data, final_weights)
    test_predictions = logistic.predict(test_data, final_weights)

    train_accuracy = round(validate(train_predictions, train_labels), 2)
    test_accuracy = round(validate(test_predictions, test_labels), 2)
    return final_weights, train_accuracy, test_accuracy
  else:
    return find_weights(filename), None, None
  
# Validation Function:
def validate(predictions, labels):
  i, count = 0, 0
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
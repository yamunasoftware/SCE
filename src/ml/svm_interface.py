### INTERFACE IMPORTS ###

import numpy as np
import os
import ml.preprocessing as preprocessing
import ml.svm as svm

### INTERFACE FUNCTIONS ###

# Default Weights Filenames and Lookup Table:
theta_filename = "/main/models/theta.npy"
theta0_filename = "/main/models/theta0.npy"
lookup_table = {
  -1: 'Negative',
  1: 'Positive'
}

# Classification Function:
def classify(sentences):
  # Predict labels for the sentences using the trained model (weights):
  thetas = train()
  feature_matrix = preprocessing.clean_set(sentences)
  predicted_labels = svm.predict(feature_matrix, thetas[0], thetas[1])
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

# Train Function:
def train():
  if find_thetas(theta_filename) is None or find_thetas(theta0_filename) is None:
    data, _, train_data, train_labels, test_data, test_labels = preprocessing.data_split()
    initial_theta = np.zeros(data.shape[1])
    initial_theta0 = np.zeros(1)

    C = 0.001
    step_size = 0.01
    tolerance = 0.01

    theta, theta0 = svm.optimize(train_data, train_labels, initial_theta, initial_theta0, C, step_size, tolerance)
    train_predictions = svm.predict(train_data, theta,theta0)
    test_predictions = svm.predict(test_data, theta,theta0)

    np.save(theta_filename, theta)
    np.save(theta0_filename, theta0)
    train_accuracy = round(validate(train_predictions, train_labels), 2)
    test_accuracy = round(validate(test_predictions, test_labels), 2)
    return [theta, theta0], train_accuracy, test_accuracy
  else:
    return [find_thetas(theta_filename), find_thetas(theta0_filename)], None, None

# Finds the Thetas:
def find_thetas(file):
  if os.path.exists(file):
    return np.load(file)
  else:
    return None
  
# Validation Function:
def validate(predictions, labels):
  i, count = 0, 0
  while i < len(predictions):
    if predictions[i] == labels[i]:
      count += 1
    i += 1
  return (count / len(predictions)) * 100.0
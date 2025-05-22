### INTERFACE IMPORTS ###

import numpy as np
import warnings
import os

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from ml import preprocessing
from ml import svm

### INTERFACE SETUP ###

# Warnings Setup:
warnings.filterwarnings('ignore')
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default File Names:
theta_filename = os.path.join(parent_dir, "theta.npy")
theta0_filename = os.path.join(parent_dir, "theta0.npy")

### INTERFACE FUNCTIONS ###

# Default Lookup Table:
lookup_table = {
  -1: 'Negative',
  1: 'Positive'
}

# Classification Steps:
def classify(sentences):
  # Load the Weights:
  thetas, message = train()
  feature_matrix = preprocessing.clean_set(sentences)

  # Predict labels for the sentences using the trained model
  predicted_labels = svm.predict(feature_matrix, thetas[0], thetas[1])
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

# Train Function:
def train():
  if find_thetas(theta_filename) is None or find_thetas(theta0_filename) is None:
    data, labels, train_data, train_labels, test_data, test_labels = preprocessing.data_split()
    initial_theta = np.zeros(data.shape[1])
    initial_theta0 = np.zeros(1)

    C = 0.001
    step_size = 0.01
    tolerance = 0.01

    theta, theta0 = svm.adam_optimizer(train_data,train_labels, initial_theta, initial_theta0,C, step_size, tolerance)
    train_predictions = svm.predict(train_data,theta,theta0)
    test_predictions = svm.predict(test_data,theta,theta0)

    np.save(theta_filename, theta)
    np.save(theta0_filename, theta0)
    message = '200 - ' + str(round(validate(train_predictions, train_labels), 2)) + ' ' + str(round(validate(test_predictions, test_labels), 2))
    return [theta, theta0], message
  else:
    return [find_thetas(theta_filename), find_thetas(theta0_filename)], '200 - Found Weights'

# Finds the Thetas:
def find_thetas(file):
  if os.path.exists(file):
    return np.load(file)
  else:
    return None
  
# Validation Function:
def validate(predictions, labels):
  count = 0
  i = 0

  while i < len(predictions):
    if predictions[i] == labels[i]:
      count += 1
    i += 1
  return (count / len(predictions)) * 100.0
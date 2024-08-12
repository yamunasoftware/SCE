### CLASSIFICATION IMPORTS ###

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from ml import preprocessing
from ml import model
from ml import training

### CLASSIFICATION FUNCTIONALITY ###

# Default Lookup Table:
lookup_table = {
  -1: 'Negative',
  1: 'Positive'
}

# Classification Steps:
def classify(sentences):
  # Load the Weights:
  weights, message = training.train()
  feature_matrix = preprocessing.clean_set(sentences)

  # Predict labels for the sentences using the trained model
  predicted_labels = model.predict(feature_matrix, weights)
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
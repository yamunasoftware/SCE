### API IMPORTS ###

import ml.logistic_interface as logistic_interface
import ml.svm_interface as svm_interface
import ml.preprocessing as preprocessing
import logs

from flask import Flask, jsonify, request
from waitress import serve
import pandas as pd

### API FUNCTIONALITY ###

# Setup Flask:
api = Flask(__name__)
@api.before_request
def before_request():
  log_identifier = 'Method: ' + request.method + '\nRoute: ' + request.path
  logs.log_info(log_identifier)

# Logistic Train API Endpoint:
@api.route('/logistic_train', methods=['GET'])
def logistic_train():
  try:
    weights, train_accuracy, test_accuracy = logistic_interface.train()
    weights = pd.Series(weights).to_json(orient='values')
    return jsonify(
      {
        'weights': weights,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
      }
    ), 200
  
  except Exception as e:
    logs.log_error('Error in /logistic_train: ' + str(e))
    return 'Server Error', 500

# SVM Train API Endpoint:
@api.route('/svm_train', methods=['GET'])
def svm_train():
  try:
    thetas, train_accuracy, test_accuracy = svm_interface.train()
    thetas = pd.Series(thetas).to_json(orient='values')
    return jsonify(
      {
        'weights': thetas,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
      }
    ), 200

  except Exception as e:
    logs.log_error('Error in /svm_train: ' + str(e))
    return 'Server Error', 500

# Logistic Classify POST Endpoint:
@api.route('/logistic_classify', methods=['POST'])
def logistic_classify():
  try:
    if not request.is_json:
      logs.log_error('Request Not JSON')
      return 'Not Acceptable', 406

    data = request.get_json()
    if type(data) is not list:
      logs.log_error('Request JSON Not list')
      return 'Not Acceptable', 406

    predictions = logistic_interface.classify(data)
    labels = logistic_interface.label_output(predictions, lookup=None)
    predictions = pd.Series(predictions).to_json(orient='values')
    labels = pd.Series(labels).to_json(orient='values')
    return jsonify(
      {
        'predictions': predictions,
        'labels': labels
      }
    ), 200

  except Exception as e:
    logs.log_error('Error in /logistic_classify: ' + str(e))
    return 'Server Error', 500

# SVM Classify POST Endpoint:
@api.route('/svm_classify', methods=['POST'])
def svm_classify():
  try:
    if not request.is_json:
      logs.log_error('Request Not JSON')
      return 'Not Acceptable', 406

    data = request.get_json()
    if type(data) is not list:
      logs.log_error('Request JSON Not list')
      return 'Not Acceptable', 406

    predictions = svm_interface.classify(data)
    labels = svm_interface.label_output(predictions, lookup=None)
    predictions = pd.Series(predictions).to_json(orient='values')
    labels = pd.Series(labels).to_json(orient='values')
    return jsonify(
      {
        'predictions': predictions,
        'labels': labels
      }
    ), 200

  except Exception as e:
    logs.log_error('Error in /svm_classify: ' + str(e))
    return 'Server Error', 500

# Health Check API Endpoint:
@api.route('/health', methods=['GET'])
def health():
  try:
    if preprocessing.validate_processing(preprocessing.data_split()):
      return 'OK', 200
    else:
      return 'Service Unavailable', 503

  except Exception as e:
    logs.log_error('Error in /health: ' + str(e))
    return 'Server Error', 500

# Runs the API:
if __name__ == '__main__':
  logs.setup_logging()
  serve(api, host="0.0.0.0", port=2016)
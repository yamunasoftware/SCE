### API IMPORTS ###

import ml.logistic_interface as logistic_interface
import ml.svm_interface as svm_interface
import ml.preprocessing as preprocessing

from flask import Flask, jsonify, request
from waitress import serve
import pandas as pd
import logging

### API SETUP ###

# Logging Setup:
file_name = 'server.log'
logging.basicConfig(
  filename = file_name,
  level = logging.INFO,
  format = '%(asctime)s %(message)s'
)

# Clear Log Function:
def clear_log():
  with open(file_name, 'w') as file:
    pass
    file.write('--- SERVER LOG ---\n\n')

# Setup Flask:
api = Flask(__name__)

# Log API Call Before:
@api.before_request
def before_request():
  logging.info(request.method + ' ' + request.path + ' ' + str(request.headers))

### API FUNCTIONALITY ###

# Logistic Classify POST Endpoint:
@api.route('/logistic_classify', methods=['POST'])
def logistic_classify():
  if not request.is_json:
    return jsonify(
      {
        'message': '406 - Not Acceptable'
      }
    )

  data = request.get_json()
  if type(data) is not list:
    return jsonify(
      {
        'message': '406 - Not Acceptable'
      }
    )

  predictions, message = logistic_interface.classify(data)
  labels = logistic_interface.label_output(predictions, lookup=None)

  predictions = pd.Series(predictions).to_json(orient='values')
  labels = pd.Series(labels).to_json(orient='values')

  return jsonify(
    {
      'message': message,
      'predictions': predictions,
      'labels': labels
    }
  )

# Logistic Train API Endpoint:
@api.route('/logistic_train', methods=['GET'])
def logistic_train():
  weights, message = logistic_interface.train()
  weights = pd.Series(weights).to_json(orient='values')

  return jsonify(
    {
      'message': message,
      'weights': weights
    }
  )

# SVM Classify POST Endpoint:
@api.route('/svm_classify', methods=['POST'])
def svm_classify():
  if not request.is_json:
    return jsonify(
      {
        'message': '406 - Not Acceptable'
      }
    )

  data = request.get_json()
  if type(data) is not list:
    return jsonify(
      {
        'message': '406 - Not Acceptable'
      }
    )

  predictions, message = svm_interface.classify(data)
  labels = svm_interface.label_output(predictions, lookup=None)

  predictions = pd.Series(predictions).to_json(orient='values')
  labels = pd.Series(labels).to_json(orient='values')

  return jsonify(
    {
      'message': message,
      'predictions': predictions,
      'labels': labels
    }
  )

# SVM Train API Endpoint:
@api.route('/svm_train', methods=['GET'])
def svm_train():
  thetas, message = logistic_interface.train()
  thetas = pd.Series(thetas).to_json(orient='values')

  return jsonify(
    {
      'message': message,
      'thetas': thetas
    }
  )

# Data Check API Endpoint:
@api.route('/data_check', methods=['GET'])
def data_check():
  message = '' 
  if preprocessing.validate_processing(preprocessing.data_split()):
    message = '200 - OK'
  else:
    message = '500 - Error'

  return jsonify(
    {
      'message': message
    }
  )

# Heartbeat API Endpoint:
@api.route('/heartbeat', methods=['GET'])
def heartbeat():
  message = '200 - OK'
  return jsonify(
    {
      'message': message
    }
  )

# Runs the API:
if __name__ == '__main__':
  serve(api, host="0.0.0.0", port=2016)
### API IMPORTS ###

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from ml import classification
from ml import training
from ml import preprocessing

from flask import Flask, jsonify, request
import pandas as pd
import logging

### API SETUP ###

# Logging Setup:
file_name = 'log.txt'
logging.basicConfig(
  filename = file_name,
  level = logging.INFO,
  format = '%(asctime)s %(message)s'
)

# Clear Log Function:
def clear_log():
  with open(file_name, 'w') as file:
    pass
    file.write('Server API Call Information:\n\n')

# Setup Flask:
api = Flask(__name__)

# Log API Call Before:
@api.before_request
def before_request():
  logging.info(request.method + ' ' + request.path + ' ' + str(request.headers))

### API FUNCTIONALITY ###

# Classification POST Endpoint:
@api.route('/classify', methods=['POST'])
def classify():
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

  predictions, message = classification.classify(data)
  labels = classification.label_output(predictions, lookup=None)

  predictions = pd.Series(predictions).to_json(orient='values')
  labels = pd.Series(labels).to_json(orient='values')

  return jsonify(
    {
      'message': message,
      'predictions': predictions,
      'labels': labels
    }
  )

# Training API Endpoint:
@api.route('/train', methods=['GET'])
def train():
  weights, message = training.train()
  weights = pd.Series(weights).to_json(orient='values')

  return jsonify(
    {
      'message': message,
      'weights': weights
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
  api.run()
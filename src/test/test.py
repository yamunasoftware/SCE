import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import sqlite3
import unittest
from datetime import date

from ml import preprocessing
from ml import svm_interface
from ml import logistic_interface

### SYSTEM TESTING SETUP ###

# Connect to SQL Database:
conn = sqlite3.connect('test.db')
c = conn.cursor()

# Create New Table:
c.execute('''
CREATE TABLE IF NOT EXISTS main (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  created DATE NOT NULL
)
''')

# Remove Existing Data:
c.execute('DELETE FROM main')
c.execute('DELETE FROM sqlite_sequence WHERE name="main"')

# Generated Test Data:
content = [
  ('I love you.', date(2024, 1, 1)),
  ('I hate you.', date(2024, 1, 2)),
  ('You are the best.', date(2024, 1, 3)),
  ('You are the worst.', date(2024, 1, 4))
]

# Insert Data into Table:
c.executemany('INSERT INTO main (content, created) VALUES (?, ?)', content)
conn.commit()

### SYSTEM TESTING EXECUTION ###

# Fetch Data:
c.execute('SELECT content FROM main ORDER BY id')
test_data = c.fetchall()
conn.close()

# Machine Learning Testing:
data = [entry[0] for entry in test_data]
logistic_predictions, logistic_message = logistic_interface.classify(data)
svm_predictions, svm_message = svm_interface.classify(data)

# Testing Labels:
logistic_labels = logistic_interface.label_output(logistic_predictions, lookup=None)
svm_labels = svm_interface.label_output(svm_predictions, lookup=None)

# System Testing Setup:
correct_predictions = [1, -1, 1, -1]
correct_labels = ['Positive', 'Negative', 'Positive', 'Negative']

### SYSTEM UNIT TESTING ###

# Unit Testing Verification Class:
class TestSCE(unittest.TestCase):
  # Tests Outputs Exist:
  def test_outputs_exist(self):
    self.assertIsNotNone(logistic_predictions)
    self.assertIsNotNone(logistic_labels)

    self.assertIsNotNone(svm_predictions)
    self.assertIsNotNone(svm_labels)
    self.assertTrue(preprocessing.validate_processing(preprocessing.data_split()))
  
  # Tests Predictions Correctness:
  def test_predictions(self):
    self.assertEqual(len(logistic_predictions), len(correct_predictions))
    if len(logistic_predictions) == len(correct_predictions):
      i = 0
      while i < len(logistic_predictions):
        self.assertEqual(logistic_predictions[i], correct_predictions[i])
        i += 1
    
    self.assertEqual(len(svm_predictions), len(correct_predictions))
    if len(svm_predictions) == len(correct_predictions):
      i = 0
      while i < len(svm_predictions):
        self.assertEqual(svm_predictions[i], correct_predictions[i])
        i += 1

  # Tests Labels Correctness:
  def test_labels(self):
    self.assertEqual(len(logistic_labels), len(correct_labels))
    if len(logistic_labels) == len(correct_labels):
      i = 0
      while i < len(logistic_labels):
        self.assertEqual(logistic_labels[i], correct_labels[i])
        i += 1

    self.assertEqual(len(svm_labels), len(correct_labels))
    if len(svm_labels) == len(correct_labels):
      i = 0
      while i < len(svm_labels):
        self.assertEqual(svm_labels[i], correct_labels[i])
        i += 1

  # Test Server Message:
  def test_message(self):
    self.assertIsNotNone(logistic_message)
    self.assertIn('200', logistic_message)

    self.assertIsNotNone(svm_message)
    self.assertIn('200', svm_message)

# Runs the Unit Tests:
if __name__ == '__main__':
  unittest.main()
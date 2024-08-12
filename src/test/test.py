import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import sqlite3
import unittest

import subprocess
from datetime import date

from ml import classification
from ml import preprocessing

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
predictions, message = classification.classify(data)

# Training Status:
labels = classification.label_output(predictions, lookup=None)
print('Status: ' + message[6:])

# System Testing Setup:
correct_predictions = [1, -1, 1, -1]
correct_labels = ['Positive', 'Negative', 'Positive', 'Negative']

### SYSTEM UNIT TESTING ###

# Unit Testing Verification Class:
class TestSCE(unittest.TestCase):
  # Tests Outputs Exist:
  def test_outputs_exist(self):
    self.assertIsNotNone(predictions)
    self.assertIsNotNone(labels)
    self.assertTrue(preprocessing.validate_processing(preprocessing.data_split()))
  
  # Tests Predictions Correctness:
  def test_predictions(self):
    self.assertEqual(len(predictions), len(correct_predictions))
    if len(predictions) == len(correct_predictions):
      i = 0
      while i < len(predictions):
        self.assertEqual(predictions[i], correct_predictions[i])
        i += 1

  # Tests Labels Correctness:
  def test_labels(self):
    self.assertEqual(len(labels), len(correct_labels))
    if len(labels) == len(correct_labels):
      i = 0
      while i < len(labels):
        self.assertEqual(labels[i], correct_labels[i])
        i += 1

  # Test Server Message:
  def test_message(self):
    self.assertIsNotNone(message)
    self.assertIn('200', message)

# Runs the Unit Tests:
if __name__ == '__main__':
  unittest.main()
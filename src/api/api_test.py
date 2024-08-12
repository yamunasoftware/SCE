import unittest
import subprocess
import requests
import time

# Unit Testing Verification Class:
class TestAPI(unittest.TestCase):
  ### SERVER TESTING SETUP ###

  # Server Node Setup:
  @classmethod
  def setUpClass(cls):
    cls.process = subprocess.Popen(["python", "serve.py"], shell=True)
    cls.wait_for_connection()

  # Wait for Server Connection:
  @classmethod
  def wait_for_connection(cls):
    timeout = 10
    interval = 0.5

    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
      if requests.get('http://127.0.0.1:5000/heartbeat').status_code == 200:
        return
      time.sleep(interval)
    raise RuntimeError('Connection Failed.')

  # Server Node Teardown:
  @classmethod
  def tearDownClass(cls):
    cls.process.terminate()
    cls.process.wait()

  ### API TESTING ###
  
  # Test Data Check:
  def test_data_check(self):
    response = requests.get('http://127.0.0.1:5000/data_check')
    self.assertEqual(response.status_code, 200)
    json_object = response.json()

    message = json_object['message']
    self.assertIn('200', message)

  # Test Train:
  def test_train(self):
    response = requests.get('http://127.0.0.1:5000/train')
    self.assertEqual(response.status_code, 200)
    json_object = response.json()

    message = json_object['message']
    self.assertIn('200', message)

  # Test Heartbeat:
  def test_heartbeat(self):
    response = requests.get('http://127.0.0.1:5000/heartbeat')
    self.assertEqual(response.status_code, 200)
    json_object = response.json()

    message = json_object['message']
    self.assertIn('200', message)

  # Test Classify:
  def test_classify(self):
    sentences = ['I hate you!']
    response = requests.post(url='http://127.0.0.1:5000/classify', json=sentences)

    self.assertEqual(response.status_code, 200)
    json_object = response.json()

    message = json_object['message']
    self.assertIn('200', message)

# Runs the Unit Tests:
if __name__ == '__main__':
  unittest.main()
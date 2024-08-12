### PREPROCESSING IMPORTS ###

import os
import string
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

### PREPROCESSING SETUP ###

# Get the parent directory of the current directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = os.path.join(parent_dir, "dataset.txt")

# Removal of Words Sets:
punctuation = list(string.punctuation)
digits = [str(x) for x in range(10)]
stops = set(['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of', 'it', 'from'])

# Vectorization Object:
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 4500)

### TRANSFORM FUNCTIONS ###

# Character Removal:
def removal(array, set):
  local_array = array
  for item in set:
    local_array = local_array.replace(item, ' ')
  return local_array

# Converts to Integers:
def transform_ints(array):
  local_array = array
  i = 0

  while i < len(array):
    local_array[i] = int(local_array[i])
    i += 1
  return local_array

# Converts Labels from 0 to -1:
def transform_labels(array):
  local_array = array
  i = 0

  while i < len(array):
    if local_array[i] == 0:
      local_array[i] = -1
    i += 1
  return local_array

### DATA FUNCTIONS ###

# Training/Test Split:
def data_split():
  # Gets Cleaned Data:
  data, labels = clean_data()
  np.random.seed(0)

  # Sets Test and Train Indices:
  test_inds = test_data_finder(labels, 0.02)
  train_inds = list(set(range(len(labels))) - set(test_inds))

  # Gets Train Data:
  train_data = data[train_inds, :]
  train_labels = [labels[i] for i in train_inds]

  # Gets Test Data:
  test_data = data[test_inds, :]
  test_labels = [labels[i] for i in test_inds]

  # Returns Test and Train Data:
  return data, labels, train_data, train_labels, test_data, test_labels

# Validate Data Processing:
def validate_processing(*argv):
  for arg in argv:
    if arg is None:
      return False
  return True

# Test Data Finder:
def test_data_finder(data, prop):
  num = len(data) * prop
  negative = []
  positive = []

  indices = []
  prev_negative = []
  prev_positive = []

  i = 0
  j = 0
  k = 0

  while i < len(data):
    if data[i] == -1:
      negative.append(i)
    elif data[i] == 1:
      positive.append(i)
    i += 1

  while j < num:
    random_index = random.choice([x for x in range(len(negative)) if x not in prev_negative])
    prev_negative.append(random_index)
    indices.append(negative[random_index])
    j += 1
  
  while k < num:
    random_index = random.choice([x for x in range(len(positive)) if x not in prev_positive])
    prev_positive.append(random_index)
    indices.append(positive[random_index])
    k += 1

  return indices

# Cleaning Data:
def clean_data():
  # Opens Training Dataset:
  with open(file_path, "r") as f:
    content = f.readlines()
  content = [line.strip() for line in content]

  # Splits Content from Labels:
  sentences = [line.split("\t")[0] for line in content]
  labels = [line.split("\t")[1] for line in content]

  # Dataset Conversions:
  labels = transform_ints(labels)
  labels = transform_labels(labels)

  # Sentence Cleaning:
  digit_sentences = [removal(sentence, digits) for sentence in sentences]
  punctuation_sentences = [removal(sentence, punctuation) for sentence in digit_sentences]
  lowercase_sentences = [sentence.lower() for sentence in punctuation_sentences]
  split_sentences = [sentence.split() for sentence in lowercase_sentences]
  processed_sentences = [" ".join(list(filter(lambda a: a not in stops, sentence))) for sentence in split_sentences]

  # Sentence Vectorization:
  vectorizer.fit(processed_sentences)
  features = vectorizer.transform(processed_sentences).toarray()

  # Returns the Feature Matrix:
  ones_column = np.ones((features.shape[0], 1))
  return np.hstack((ones_column, features)), labels

# Clean Set Data Inputs:
def clean_set(sentences):
  digit_sentences = [removal(sentence, digits) for sentence in sentences]
  punctuation_sentences = [removal(sentence, punctuation) for sentence in digit_sentences]
  lowercase_sentences = [sentence.lower() for sentence in punctuation_sentences]
  split_sentences = [sentence.split() for sentence in lowercase_sentences]
  processed_sentences = [" ".join(list(filter(lambda a: a not in stops, sentence))) for sentence in split_sentences]

  features = vectorizer.transform(processed_sentences).toarray()
  ones_column = np.ones((features.shape[0], 1))
  return np.hstack((ones_column, features))
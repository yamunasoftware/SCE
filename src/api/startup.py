### API STARTUP SCRIPT ###

import os
import ml.logistic_interface as logistic_interface
import ml.svm_interface as svm_interface

def train_models():
  if not os.path.exists('/main/models/weights.npy'):
    svm_interface.train()
  if not os.path.exists('/main/models/theta.npy') or not os.path.exists('/main/models/theta0.npy'):
    logistic_interface.train()

train_models()
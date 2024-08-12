import numpy as np

def weight_derivative(weights, feature_matrix, labels):
  derivative = np.zeros_like(weights)
  for i in range(feature_matrix.shape[0]):
    xi = feature_matrix[i]
    yi = labels[i]
    derivative -= yi * xi / (1 + np.exp(yi * np.dot(weights.T, xi)))
  return derivative

def gradient_descent(feature_matrix, labels, initial_weights, step_size, tolerance):
  converged = False 
  weights = np.array(initial_weights)
  i = 0
  while not converged:
    i += 1
    gradient = weight_derivative(weights, feature_matrix, labels)
    weights -= step_size * gradient
    gradient_magnitude = np.linalg.norm(gradient)
    
    if gradient_magnitude < tolerance:
      converged = True        
  return(weights)

def predict(feature_matrix, weights):
  combination = np.dot(feature_matrix, weights)
  probability = 1 / (1 + np.exp(-combination))
  labels = np.where(probability >= 0.5, 1, -1)
  return labels
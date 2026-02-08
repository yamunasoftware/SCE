import numpy as np

# Prediction Function:
def predict(feature_matrix, theta, theta0):
  return ((feature_matrix.dot(theta) + theta0) >= 0)*2-1

# Finds the Weights Derivatives:
def weight_derivative(theta, theta0, C, feature_matrix, labels):
  n, d = feature_matrix.shape
  grad_theta = np.zeros(d)
  grad_theta0 = 0

  for i in range(n):
    x_i = feature_matrix[i]
    y_i = labels[i]
    t = y_i * (np.dot(theta, x_i) + theta0)

    if t <= 0:
      derivative_theta = -y_i * x_i
      derivative_theta_0 = -y_i
    elif t < 1:
      derivative_theta = -y_i * x_i * (1 - t)
      derivative_theta_0 = -y_i * (1 - t)
    else:
      derivative_theta = np.zeros_like(theta)
      derivative_theta_0 = 0

    grad_theta += derivative_theta
    grad_theta0 += derivative_theta_0
  grad_theta += 2 * theta
  return grad_theta*C, grad_theta0*C

# Optimization Training Function:
def optimize(feature_matrix, labels, initial_theta, initial_theta0, C, step_size, tolerance):
  b1 = 0.9
  b2 = 0.999
  eps = 10**-8
  
  converged = False
  i = 0

  m = np.zeros(len(initial_theta))
  v = np.zeros(len(initial_theta))

  m0 = np.zeros(1)
  v0 = np.zeros(1)

  theta = np.array(initial_theta)
  theta0 = np.array(initial_theta0)

  while not converged:
    i += 1
    grad_theta, grad_theta0  = weight_derivative(theta, theta0, C,feature_matrix, labels)
    
    m = (1 - b1) * grad_theta      + b1 * m
    v = (1 - b2) * (grad_theta**2) + b2 * v
    mhat = m / (1 - b1**(i + 1))
    vhat = v / (1 - b2**(i + 1))
    theta = theta - step_size*mhat/(np.sqrt(vhat) + eps)
    
    m0 = (1 - b1) * grad_theta0      + b1 * m0
    v0 = (1 - b2) * (grad_theta0**2) + b2 * v0
    mhat0 = m0 / (1 - b1**(i + 1))
    vhat0 = v0 / (1 - b2**(i + 1))
    theta0 = theta0 - step_size*mhat0/(np.sqrt(vhat0) + eps)
    
    gradient_magnitude = np.sqrt(np.sum(grad_theta**2))
    if gradient_magnitude < tolerance:
      converged = True
  return theta, theta0
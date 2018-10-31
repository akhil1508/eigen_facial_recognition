import numpy as np
from scipy.spatial import distance
"""
  Computes the dot product of the given eigenvector and the matrix of means
"""
def return_weight_vector(eigenvector, mean_matrix) :
  #print (np.dot(eigenvector,mean_matrix))
  return np.dot(eigenvector, mean_matrix)

"""
  Computes the 'distance' between the test_weight and the given weights
"""
def distance_(test_weight, weights) :
 # print(x[0])
#  print("Weights, test weights", weights, test_weight)
  dist = np.array([])
  print(test_weight.shape, weights.shape)
  for index, weight in enumerate(weights) :
      dist = np.append(dist, np.linalg.norm((test_weight-weight)))
  return np.linalg.norm(dist)
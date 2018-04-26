import numpy as np
from scipy.spatial import distance
def return_weight_vector(eigenvector, mean_matrix) :
  #print (np.dot(eigenvector,mean_matrix))
  return np.dot(eigenvector, mean_matrix)

def distance_(test_weight, weights) :
 # print(x[0])
#  print("Weights, test weights", weights, test_weight)
  dist = np.array([])
  print(test_weight.shape, weights.shape)
  for index, weight in enumerate(weights) :
      dist = np.append(dist, np.linalg.norm((test_weight-weight)))
  return np.linalg.norm(dist)
import numpy as np

def mean_across_columns(array_of_images) :
  averages_array = np.array([])
  for index, col in enumerate(array_of_images.T) :
    avg = np.mean(col)
    averages_array = np.append(averages_array, avg)
  return averages_array

def subtract_mean(array_of_images) :
  return array_of_images - mean_across_columns(array_of_images).T

def find_covariance(array_of_images) :
  array_of_images = subtract_mean(array_of_images)
  return np.cov(array_of_images)
import numpy as np

def mean_across_columns(array_of_images) :
  averages_array = np.array([])
#  print(averages_array)
 # print(array_of_images.shape)
  for index, col in enumerate(array_of_images.T) :
    avg = np.mean(col)
    #print(avg)
    averages_array = np.append(averages_array, avg)
  return averages_array

def subtract_mean(array_of_images) :
  #print("Hi")
  #rint(mean_across_columns(array_of_images).shape)
  return array_of_images - mean_across_columns(array_of_images).T

def find_covariance(array_of_images) :
  array_of_images = subtract_mean(array_of_images)
  return np.cov(array_of_images)
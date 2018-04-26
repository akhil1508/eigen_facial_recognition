import numpy as np
from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean

def load_grayscale_image(url) :
  return color.rgb2grey(io.imread(url))
  
def rescale_image(image, factor) :
  return rescale(image, factor)

def preprocess_image(url, factor = 0.25) :
  return rescale_image(load_grayscale_image(url), factor)
  #return load_grayscale_image(url)
  
def preprocess_images(urls, factor = 0.25) :
  for index, url in enumerate(urls) :
    image = preprocess_image(url, factor) 
    if index == 0 :
      array_of_images = np.zeros((len(urls), image.shape[0], image.shape[1]))
    array_of_images[index] = image 
  return array_of_images

def preprocess_images_and_flatten(urls, factor = 0.25) :
  for index, url in enumerate(urls) :
    image = preprocess_image(url, factor)
    if index == 0 :
      cols = image.shape[0] * image.shape[1]
      array_of_images = np.zeros((len(urls), cols))
    array_of_images[index] = image.flatten()
  return array_of_images
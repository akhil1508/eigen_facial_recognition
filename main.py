import preprocessor
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
urls = ["./dataset/paavan_1.jpg", "./dataset/paavan_2.jpg","./dataset/paavan_3.jpg", "./dataset/paavan_4.jpg", "./dataset/paavan_5.jpg"]
import covariance
import jacobi

'''
fig, axes = plt.subplots(nrows= len(urls))
images = preprocessor.preprocess_images(urls)
for index, image in enumerate(images) :
  axes[index].imshow(images[index])
  axes[index].set_title("Image " + str(index+1))
plt.show()
'''
images = preprocessor.preprocess_images_and_flatten(urls, factor=0.0625/4.0)
print(images.shape)
covariance_matrix = covariance.find_covariance(images.T)

print(jacobi.jacobi(covariance_matrix))
'''
reduce dimensionality by extracting the smallest number components that account for most of the variation in the original data. By doing so, we'd get get rid of the redundancy and preserve the variance in a smaller number of coefficients.'''

print(covariance_matrix)
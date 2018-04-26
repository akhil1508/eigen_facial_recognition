import preprocessor
import numpy as np
import warnings
import covariance
from skimage import io

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
akhil_urls = ["./dataset/akhil_1.jpg", "./dataset/akhil_2.jpg","./dataset/akhil_3.jpg","./dataset/akhil_4.jpg","./dataset/akhil_5.jpg","./dataset/akhil_6.jpg","./dataset/akhil_7.jpg","./dataset/akhil_8.jpg","./dataset/akhil_9.jpg","./dataset/akhil_11.jpg","./dataset/akhil_12.jpg","./dataset/akhil_13.jpg","./dataset/akhil_14.jpg","./dataset/akhil_15.jpg","./dataset/akhil_16.jpg","./dataset/akhil_17.jpg","./dataset/akhil_18.jpg","./dataset/akhil_19.jpg","./dataset/akhil_20.jpg"]
akhil_test_url = "./dataset/akhil_10.jpg"

paavan_urls = ["./dataset/paavan_1.jpg", "./dataset/paavan_2.jpg", "./dataset/paavan_3.jpg", "./dataset/paavan_4.jpg","./dataset/paavan_5.jpg"]
paavan_test_url = "./dataset/paavan_6.jpg"
import sklearn.decomposition as decomp
import weights as ws
# akhil
images = preprocessor.preprocess_images_and_flatten(akhil_urls, factor=0.0625)  
akhil_mean = covariance.mean_across_columns(images)
akhil_mean_matrix = covariance.subtract_mean(images)
akhil_weights = np.zeros((5,len(akhil_urls)))
pca = decomp.PCA(n_components = 5)
images = pca.fit_transform(images.T).T
variance_ratio = pca.explained_variance_ratio_

akhil_test_image = preprocessor.preprocess_image(akhil_test_url, 0.0625).flatten()
fig, axes = plt.subplots(nrows =4, ncols=2)
axes = axes.ravel()
for x in range(0, 5) :
  akhil_weights[x] = ws.return_weight_vector(images[x], akhil_mean_matrix.T)
axes[0].imshow(images[0].reshape((80,60)))
axes[0].set_title("Akhil Eigenvector image 1")
axes[2].imshow(images[1].reshape((80,60)))
axes[4].imshow(images[2].reshape((80,60)))

axes[4].set_title("Akhil Eigenvector image 2")
axes[6].imshow(images[3].reshape((80,60)))
axes[6].set_title("Akhil Eigenvector image 3")
akhil_test_weight = ws.return_weight_vector(akhil_test_image, akhil_mean_matrix.T)

paavan_test_image = preprocessor.preprocess_image(paavan_test_url, 0.0625).flatten()
images = preprocessor.preprocess_images_and_flatten(paavan_urls, factor=0.0625)  
paavan_mean = covariance.mean_across_columns(images)
paavan_mean_matrix = covariance.subtract_mean(images)
paavan_weights = np.zeros((3,len(paavan_urls)))
pca = decomp.PCA(n_components =3)
images = pca.fit_transform(images.T).T
variance_ratio = pca.explained_variance_ratio_
paavan_akhil_test_image = paavan_test_image - akhil_mean
akhil_paavan_test_image = akhil_test_image - paavan_mean
akhil_test_image -= akhil_mean
paavan_test_image -= paavan_mean
for x in range(0, 3) :
  paavan_weights[x] = ws.return_weight_vector(images[x], paavan_mean_matrix.T)
axes[1].imshow(images[0].reshape((80,60)))
axes[1].set_title("Paavan Eigenvector image 1")
axes[3].imshow(images[1].reshape((80,60)))
axes[5].imshow(images[2].reshape((80,60)))

axes[5].set_title("Paavan Eigenvector image 3")
axes[3].set_title("Paavan Eigenvector image 2")
paavan_akhil_test_weight = ws.return_weight_vector(paavan_akhil_test_image, akhil_mean_matrix.T)

akhil_paavan_test_weight = ws.return_weight_vector(akhil_paavan_test_image, paavan_mean_matrix.T)
paavan_test_weight = ws.return_weight_vector(paavan_test_image, paavan_mean_matrix.T)

fig2 = plt.figure()
a=fig2.add_subplot(1,2,1)
b = fig2.add_subplot(1,2,2)
paavan_image = io.imread(paavan_test_url)
akhil_image= io.imread(akhil_test_url)
if(ws.distance_(akhil_test_weight, akhil_weights) <= ws.distance_(paavan_akhil_test_weight, akhil_weights)) :
  a.imshow(akhil_image)
  a.set_title("This is Akhil")
else :
  a.imshow(paavan_image)
  a.set_title("This is Akhil")

if(ws.distance_(paavan_test_weight, paavan_weights) <= ws.distance_(akhil_paavan_test_weight, paavan_weights)) :
  b.imshow(paavan_image)
  b.set_title("This is Paavan")
else :
  b.imshow(akhil_image)
  b.set_title("This is Paavan")


plt.show()
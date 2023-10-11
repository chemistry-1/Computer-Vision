import cv2
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
import matplotlib.pyplot as plt

#Direktori gambar. Silakan menggunakan gambar apapun.
dir = "suzume x daijin.jpg"
colors = ("r","g","b")

img_opencv= cv2.imread(dir) #Membuka gambar dengan OpenCV
img_skimage= io.imread(dir) #Membuka gambar dengan Skimage

fig, ax =plt.subplots(2, 1)
imgplot=ax[0].imshow(img_opencv)
ax[0].set_title("Image loaded with OpenCV")
imgplot2=ax[1].imshow(img_skimage)
ax[1].set_title("Image loaded with Scikit Image")
plt.show()
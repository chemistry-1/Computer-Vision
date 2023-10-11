import cv2
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
import matplotlib.pyplot as plt

#Direktori gambar. Silakan menggunakan gambar apapun.
dir = "suzume & daijin.jpg"
url = "https://i.pinimg.com/564x/bb/3d/b5/bb3db59b10d10c69f34f3d51356ad019.jpg"
colors = ("r","g","b")

img_opencv = cv2.imread(dir)
img_skimage= io.imread(url) #Membuka gambar dengan Skimage

# img_skimage = img_skimage[:, :, [2,1,0]]
img_skimage = img_skimage[:, :, ::-1]
# img_opencv = img_opencv[:, :, [2,1,0]]
# img_opencv = img_opencv[:, :, ::-1]

print("The shape of the image is ", img_opencv.shape)
height, width, color = img_opencv.shape

def main():
    fig, ax =plt.subplots(2, 2)
    imgplot=ax[0,0].imshow(img_opencv)
    ax[0,0].set_title("Image loaded with OpenCV")
    imgplot2=ax[0,1].imshow(img_skimage)
    ax[0,1].set_title("Image loaded with Scikit Image")
    for i, j, k in zip(range (3), colors,range (3)):
        histogram_ocv, bin_edges_ocv = np.histogram(
        img_opencv[ :, :, i], bins=256, range=(0, 256)
    )
        histogram_skimage, bin_edges_skimage = np.histogram(
        img_skimage[:, :, k], bins=256, range=(0, 256)
    )
        ax[1,0].plot(bin_edges_ocv[0:-1], histogram_ocv, color=j)
        ax[1,1].plot(bin_edges_skimage[0:-1], histogram_skimage, color=j)
    plt.show()

if __name__== "__main__":
    main()
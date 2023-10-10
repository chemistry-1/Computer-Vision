from turtle import shape
import cv2
import numpy as np
import matplotlib.pyplot as plt

# original image
#Silakan mengganti dengan gambar anda sendiri
f_R = cv2.cvtColor(cv2.imread(r"WIN_20230126_05_35_34_Pro.jpg"), cv2.COLOR_BGR2RGB)

def main():
    #Generate a Gaussian noise
    x, y = np.shape(f_R[:,:,0])
    gauss_noise=np.zeros((x,y),dtype=np.uint8)
    cv2.randn(gauss_noise,10,10)
    gauss_noise=(gauss_noise*1).astype(np.uint8)
    
    #Generate a salt and pepper noise
    imp_noise=np.zeros((x,y),dtype=np.uint8)
    cv2.randu(imp_noise,0,255)
    impulse_noise=cv2.threshold(imp_noise,245,255,cv2.THRESH_BINARY)[1]

    #Generate uniform noise
    uni_noise=np.zeros((x,y),dtype=np.uint8)
    cv2.randu(uni_noise,0,255)
    uniform_noise=(uni_noise*0.1).astype(np.uint8)
    
    # adding a gaussian noise to imgage
    g = cv2.add(f_R[:,:,2], gauss_noise)
    g_imp = cv2.add(f_R[:,:,2], impulse_noise)
    g_uni = cv2.add(f_R[:,:,2], uniform_noise)

    
    #Display image
    fig, (ax1, ax2, ax3) =plt.subplots(1, 3)
    ax1.imshow(gauss_noise, cmap="gray")
    ax1.set_title("Gaussian noise")
    ax2.imshow(f_R[:,:,2], cmap='gray')
    ax2.set_title("Original Image")
    ax3.imshow(g, cmap='gray')
    ax3.set_title("Corrupted Image")
    fig.tight_layout()
    # plt.show()
    
    fig, (ay1, ay2, ay3) =plt.subplots(1, 3)
    ay1.imshow(imp_noise, cmap="gray")
    ay1.set_title("impulse noise")
    ay2.imshow(f_R[:,:,2], cmap='gray')
    ay2.set_title("Original Image")
    ay3.imshow(g_imp, cmap='gray')
    ay3.set_title("Corrupted Image")
    fig.tight_layout()
    # plt.show()
    
    fig, (az1, az2, az3) =plt.subplots(1, 3)
    az1.imshow(uni_noise, cmap="gray")
    az1.set_title("uniform noise")
    az2.imshow(f_R[:,:,2], cmap='gray')
    az2.set_title("Original Image")
    az3.imshow(g_uni, cmap='gray')
    az3.set_title("Corrupted Image")
    fig.tight_layout()
    plt.show()

if __name__== "__main__":
    main()
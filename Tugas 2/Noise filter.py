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
    cv2.randn(gauss_noise,100,50)
    gauss_noise=(gauss_noise*1).astype(np.uint8)
    
    #Generate a salt and pepper noise
    imp_noise=np.zeros((x,y),dtype=np.uint8)
    cv2.randu(imp_noise,0,255)
    impulse_noise=cv2.threshold(imp_noise,200,255,cv2.THRESH_BINARY)[1]

    #Generate uniform noise
    uni_noise=np.zeros((x,y),dtype=np.uint8)
    cv2.randu(uni_noise,100,255)
    uniform_noise=(uni_noise*0.1).astype(np.uint8)
    
    # adding a gaussian noise to imgage
    g = cv2.add(f_R[:,:,2], gauss_noise)
    g_imp = cv2.add(f_R[:,:,2], impulse_noise)
    g_uni = cv2.add(f_R[:,:,2], uniform_noise)

    gauss_filter(g, g)
    gauss_filter(g_imp, g_imp)
    gauss_filter(g_uni, g_uni)
    
    median_filter(g, g)
    median_filter(g_imp, g_imp)
    median_filter(g_uni, g_uni)
    
    NLM_filter(g, g)
    NLM_filter(g_imp, g_imp)
    NLM_filter(g_uni, g_uni)
    
    bil_filter(g, g)
    bil_filter(g_imp, g_imp)
    bil_filter(g_uni, g_uni)
    
    #Display image
    fig, (ax1, ax2, ax3) =plt.subplots(1, 3)
    ax1.imshow(gauss_noise, cmap="gray")
    ax1.set_title("Gaussian noise")
    ax2.imshow(f_R[:,:,2], cmap='gray')
    ax2.set_title("Original Image")
    ax3.imshow(g, cmap='gray')
    ax3.set_title("Corrupted Image")
    fig.tight_layout()
    plt.show()

def gauss_filter(noise_img, n_img):
    fig, (ax1, ax2) =plt.subplots(1, 2)
    denoised_img = cv2.GaussianBlur(noise_img, (25, 25), 0)
    ax1.imshow(denoised_img, cmap="gray")
    ax1.set_title("Gauss Filtered Image")
    ax2.imshow(n_img, cmap='gray')
    ax2.set_title("Noise Image")
    # cv2.imshow('Gaussion Blurring', denoised_img)
    # cv2.waitKey(0)

def median_filter(noise_img, n_img):
    fig, (ax1, ax2) =plt.subplots(1, 2)
    denoised_img = cv2.medianBlur(noise_img, 5)

    ax1.imshow(denoised_img, cmap="gray")
    ax1.set_title("Median Filtered Image")
    ax2.imshow(n_img, cmap='gray')
    ax2.set_title("Noise Image")

def NLM_filter(noise_img, n_img):
    fig, (ax1, ax2) =plt.subplots(1, 2)
    denoised_img = cv2.fastNlMeansDenoising(noise_img, None, 40, 40, 20)
    
    ax1.imshow(denoised_img, cmap="gray")
    ax1.set_title("NLM Filtered Image")
    ax2.imshow(n_img, cmap='gray')
    ax2.set_title("Noise Image")
    
def bil_filter(noise_img, n_img):
    fig, (ax1, ax2) =plt.subplots(1, 2)
    denoised_img = cv2.bilateralFilter(noise_img, 50, 40, 100)
    
    ax1.imshow(denoised_img, cmap="gray")
    ax1.set_title("Bilateral Filtered Image")
    ax2.imshow(n_img, cmap='gray')
    ax2.set_title("Noise Image")

if __name__== "__main__":
    main()
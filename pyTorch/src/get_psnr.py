# Calculate PSNR of HR and model output
import cv2
import numpy as np

def psnr(img1, img2):
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    # Calculate PSNR value
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

# Load the two images
img1 = cv2.imread("left.png")
img2 = cv2.imread("C:/Users/Percy/Flickr1024/Test/0035_L.png")
img3 = cv2.imread("right.png")
img4 = cv2.imread("C:/Users/Percy/Flickr1024/Test/0035_R.png")

# Convert the images to grayscale if needed
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

# Calculate PSNR value
psnr_value = (psnr(img1, img2) + psnr(img3, img4))/2

# Print the PSNR value
print("PSNR value:", psnr_value)
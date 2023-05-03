import cv2
from skimage.metrics import structural_similarity as ssim
import os

# define the directory where the images are located
img_dir = 'C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/FinalOutput'
img_dir_label = 'C:/Users/Percy/Flickr1024/Test'

# create empty lists to store the PSNR and SSIM values
psnr_values_L = []
ssim_values_L = []
psnr_values_R = []
ssim_values_R = []

# loop through each image pair in the directory
for filename in os.listdir(img_dir):
        model_img = cv2.imread(os.path.join(img_dir, filename))
        label_img = cv2.imread(os.path.join(img_dir_label, filename))
        
        # convert the input images to the same type
        # model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2GRAY)
        # label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
        
        # compute the PSNR and SSIM values
        psnr = cv2.PSNR(model_img, label_img)
        ssim_ = ssim(model_img, label_img, channel_axis=2)
        
        # append the PSNR and SSIM values to the respective lists
        if filename.endswith('_L.png'):
            psnr_values_L.append(psnr)
            ssim_values_L.append(ssim_)
        else:
            psnr_values_R.append(psnr)
            ssim_values_R.append(ssim_)

# compute the mean PSNR and SSIM values
mean_psnr = (sum(psnr_values_L) / len(psnr_values_L) + sum(psnr_values_L) / len(psnr_values_L)) / 2
mean_ssim = (sum(ssim_values_L) / len(ssim_values_L) + sum(ssim_values_R) / len(ssim_values_R)) / 2

# print the mean PSNR and SSIM values
print("Mean PSNR: {:.2f} dB".format(mean_psnr))
print("Mean SSIM: {:.4f}".format(mean_ssim))

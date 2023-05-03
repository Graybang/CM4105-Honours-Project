import torch
import matplotlib.pyplot as plt
import time
import edsr_stereo_swin
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import tensorflow_datasets as tfds
import tensorflow as tf
import torchvision.transforms as T
import torchvision.datasets as datasets

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
# from datasets import load_dataset
from data_module_stereo import Flickr1024, RandomHorizontalFlip, RandomVerticalFlip, Normalize, ToTensor, Compose

# load the super-resolution model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = edsr_stereo_swin.edsr(2,4, 96, 3).to(device)
state_dict = torch.load("pyTorch/src/outputs_stereo/model10.pth")
model.load_state_dict(state_dict)

criterion = nn.L1Loss()

test_dir = 'C:/Users/Percy/Flickr1024/Test_patched'

transforms = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5),
                            ToTensor(),
                            ])

testset = Flickr1024(root_dir=test_dir, im_size=64, scale=2, transform=transforms)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

def psnr(label, outputs_stereo, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    label = label.cpu().detach().numpy()
    outputs_stereo = outputs_stereo.cpu().detach().numpy()
    img_diff = outputs_stereo - label
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR

def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(testset)/dataloader.batch_size)):
            image_data_L = data[0].to(device)
            image_data_R = data[1].to(device)
            label_L = data[2].to(device)
            label_R = data[3].to(device)
            
            outputs_L, outputs_R = model(image_data_L, image_data_R)
            loss = criterion(outputs_L, label_L) + criterion(outputs_R, label_R)
            # add loss of each item (total items in a batch = batch size) 
            running_loss += loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = (psnr(label_L, outputs_L) + psnr(label_R, outputs_R)) / 2
            running_psnr += batch_psnr
        outputs_L = outputs_L.cpu()
        save_image(outputs_L, f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/test_sr{epoch}_L.png")
        outputs_R = outputs_R.cpu()
        save_image(outputs_R, f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/test_sr{epoch}_R.png")
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(testset)/dataloader.batch_size)
    return final_loss, final_psnr

print(model)

val_epoch_loss, val_epoch_psnr = validate(model, testloader, 1)

print(val_epoch_loss)

print(val_epoch_psnr)
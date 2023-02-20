import torch
import matplotlib.pyplot as plt
import time
import psnr
import edsr_stereo
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import tensorflow_datasets as tfds
import tensorflow as tf
import torchvision.transforms as T

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
from datasets import load_dataset
from data_module_stereo import Flickr1024, RandomHorizontalFlip, RandomVerticalFlip, Normalize, ToTensor, Compose

train_dir = 'C:/Users/Percy/Flickr1024/Train'
test_dir = 'C:/Users/Percy/Flickr1024/Test'
val_dir = 'C:/Users/Percy/Flickr1024/Validation'

transforms = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5),
                            ToTensor(),
                            ])

trainset = Flickr1024(root_dir=train_dir, im_size=100, scale=2, transform=transforms)
validset = Flickr1024(root_dir=val_dir, im_size=100, scale=2, transform=transforms)

trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
validloader = DataLoader(validset, batch_size=8, shuffle=True)

upscale_factor = 2
resblock_layers = 16
channels = 256
kernel = 3

lowres_L, lowres_R, highres_L, highres_R = next(iter(trainloader))

# High Resolution Images
plt.figure(figsize=(10, 10))
for i in range(4):
    save_image(highres_L[i], f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/high_res{i}_L.png")
    save_image(highres_R[i], f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/high_res{i}_R.png")


# Low Resolution Images
plt.figure(figsize=(10, 10))
for i in range(4):
    save_image(lowres_L[i], f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/low_res{i}_L.png")
    save_image(lowres_R[i], f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/low_res{i}_R.png")

# initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Computation device: ', device)
model = edsr_stereo.edsr(upscale_factor,resblock_layers, channels, kernel).to(device)
print(model)

epochs = 50

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
# loss function 
criterion = nn.L1Loss()

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
    
def train(model, dataloader):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(trainset)/dataloader.batch_size)):
        image_data_L = data[0].to(device)
        image_data_R = data[1].to(device)
        label_L = data[2].to(device)
        label_R = data[3].to(device)
    
        # print(image_data_L.size())
        # print(label_L.size())
        # print(image_data_R.size())
        # print(label_R.size())

        # zero grad the optimizer
        optimizer.zero_grad()
        outputs_L, outputs_R = model(image_data_L, image_data_R)
        loss = criterion(outputs_L, label_L) + criterion(outputs_R, label_R)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        # calculate batch psnr (once every `batch_size` iterations)
        batch_psnr =  (psnr(label_L, outputs_L) + psnr(label_R, outputs_R)) / 2
        running_psnr += batch_psnr
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(trainset)/dataloader.batch_size)
    return final_loss, final_psnr

def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(validset)/dataloader.batch_size)):
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
        save_image(outputs_L, f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/val_sr{epoch}_L.png")
        outputs_R = outputs_R.cpu()
        save_image(outputs_R, f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/val_sr{epoch}_R.png")
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(validset)/dataloader.batch_size)
    return final_loss, final_psnr

train_loss, val_loss = [], []
train_psnr, val_psnr = [], []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss, train_epoch_psnr = train(model, trainloader)
    val_epoch_loss, val_epoch_psnr = validate(model, validloader, epoch)
    print(f"Train PSNR: {train_epoch_psnr:.3f}")
    print(f"Val PSNR: {val_epoch_psnr:.3f}")
    train_loss.append(train_epoch_loss)
    train_psnr.append(train_epoch_psnr)
    val_loss.append(val_epoch_loss)
    val_psnr.append(val_epoch_psnr)
end = time.time()
print(f"Finished training in: {((end-start)/60):.3f} minutes")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/loss.png')
plt.show()
# psnr plots
plt.figure(figsize=(10, 7))
plt.plot(train_psnr, color='green', label='train PSNR dB')
plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.savefig('C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/psnr.png')
plt.show()
# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), 'C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/model.pth')
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
import time
import h5py
import srcnn
import psnr
import edsr
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
from data_module import DIV2K_x2, RandomHorizontalFlip, RandomVerticalFlip, Normalize, ToTensor, Compose

train_dir = 'C:/Users/Percy/Downloads/EDSR-pytorch-master (1)/EDSR-pytorch-master/data/train'
val_dir = 'C:/Users/Percy/Downloads/EDSR-pytorch-master (1)/EDSR-pytorch-master/data/validation'

train_transforms = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5),
                            ToTensor(),
                            ])

valid_transforms = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5),
                            ToTensor(),
                            ])

trainset = DIV2K_x2(root_dir=train_dir, im_size=80, scale=2, transform=train_transforms)
validset = DIV2K_x2(root_dir=val_dir, im_size=80, scale=2, transform=valid_transforms)

trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
validloader = DataLoader(validset, batch_size=16, shuffle=True)

upscale_factor = 2
resblock_layers = 16
channels = 256
kernel = 3

lowres, highres = next(iter(trainloader))

# High Resolution Images
plt.figure(figsize=(10, 10))
for i in range(4):
    save_image(highres[i], f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs/high_res{i}.png")


# Low Resolution Images
plt.figure(figsize=(10, 10))
for i in range(4):
    save_image(lowres[i], f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs/low_res{i}.png")


# initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Computation device: ', device)
model = edsr.edsr(upscale_factor,resblock_layers, channels, kernel).to(device)
print(model)

# criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

# psnr = psnr.PSNR()

# epochs = 10
# print_every = 25
# train_loss = 0
# batch_num = 0

# for epoch_num in range(epochs):
#     for img, label in trainloader:
#         img, label = img.cuda(), label.cuda()
#         optimizer.zero_grad()
#         pred = model(img)
#         # print(pred.shape, label.shape)
#         batch_num += 1
#         loss = criterion(pred, label)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         psnr_value = psnr(pred, label)
#         # image1 = T.ToPILImage()(pred[0])
#         # image2 = T.ToPILImage()(label[0])
#         # image1.show()
#         # image2.show()
#         if batch_num % print_every == 0:
#             print('Training Loss: {:.4f}'.format(train_loss / print_every))
#             print('PSNR: {:.4f}'.format(psnr_value))

#     with torch.no_grad():
#         val_loss = 0
#         model.eval()
#         for val_ims, val_lbs in validloader:
#             val_ims, val_lbs = val_ims.cuda(), val_lbs.cuda()
#             test_pred = model(val_ims)
#             vloss = criterion(test_pred, val_lbs)
#             val_loss += vloss.item()

#         print('Epoch : {}/{}'.format(epoch_num, epochs))
#         print('Training Loss : {:.4f}'.format(train_loss / print_every))
#         print('Validation Loss: {:.4f}'.format(val_loss / len(validloader)))
#         train_loss = 0
#         model.train()

epochs = 100

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
# loss function 
criterion = nn.L1Loss()

def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
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
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        # calculate batch psnr (once every `batch_size` iterations)
        batch_psnr =  psnr(label, outputs)
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
            image_data = data[0].to(device)
            label = data[1].to(device)
            
            outputs = model(image_data)
            loss = criterion(outputs, label)
            # add loss of each item (total items in a batch = batch size) 
            running_loss += loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr
        outputs = outputs.cpu()
        save_image(outputs, f"C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs/val_sr{epoch}.png")
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
plt.savefig('C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs/loss.png')
plt.show()
# psnr plots
plt.figure(figsize=(10, 7))
plt.plot(train_psnr, color='green', label='train PSNR dB')
plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.savefig('C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs/psnr.png')
plt.show()
# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), 'C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs/model.pth')
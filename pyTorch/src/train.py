import torch
import matplotlib
import matplotlib.pyplot as plt
import time
import h5py
import srcnn
import edsr
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import tensorflow_datasets as tfds
import tensorflow as tf

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
from datasets import load_dataset

matplotlib.style.use('ggplot')

AUTOTUNE = tf.data.AUTOTUNE

# learning parameters
batch_size = 2 # batch size, reduce if facing OOM error
epochs = 200 # number of epochs to train the SRCNN model for
lr = 0.001 # the learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # input image dimensions
# img_rows, img_cols = 33, 33
# out_rws, out_cols = 132, 132

file = h5py.File('pyTorch/input/train_mscale.h5')
# `in_train` has shape (21884, 33, 33, 1) which corresponds to
# 21884 image patches of 33 pixels height & width and 1 color channel
in_train = file['data'][:] # the training data
out_train = file['label'][:] # the training labels
file.close()
# change the values to float32
in_train = in_train.astype('float32')
out_train = out_train.astype('float32')

# print(out_train)

# (x_train, x_val, y_train, y_val) = train_test_split(in_train, out_train, test_size=0.25)
# print('Training samples: ', x_train.shape[0])
# print('Validation samples: ', x_val.shape[0])

# Download DIV2K from TF Datasets
# Using bicubic 4x degradation type
div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()

# Taking train data from div2k_data object
train = div2k_data.as_dataset(split="train", as_supervised=True)
train_cache = train.cache()
# Validation data
val = div2k_data.as_dataset(split="validation", as_supervised=True)
val_cache = val.cache()

print(train_cache)
print(val_cache)

# the dataset module
class SRCNNDataset(Dataset):
    def __init__(self, lowres_img, highres_img):
        self.lowres_img = lowres_img
        self.highres_img = highres_img

    def __len__(self):
        return (len(self.lowres_img))
    
    def __getitem__(self, index):
        image = self.lowres_img[index]
        label = self.highres_img[index]
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )
    
# train and validation data
train_data = SRCNNDataset(train)
val_data = SRCNNDataset(val)
# train and validation loaders
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

def flip_left_right(lowres_img, highres_img):
    """Flips Images to left and right."""

    # Outputs random values from a uniform distribution in between 0 to 1
    rn = tf.random.uniform(shape=(), maxval=1)
    # If rn is less than 0.5 it returns original lowres_img and highres_img
    # If rn is greater than 0.5 it returns flipped image
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (
            tf.image.flip_left_right(lowres_img),
            tf.image.flip_left_right(highres_img),
        ),
    )


def random_rotate(lowres_img, highres_img):
    """Rotates Images by 90 degrees."""

    # Outputs random values from uniform distribution in between 0 to 4
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    # Here rn signifies number of times the image(s) are rotated by 90 degrees
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)


def random_crop(lowres_img, highres_img, hr_crop_size=96, scale=4):
    """Crop images.

    low resolution images: 24x24
    high resolution images: 96x96
    """
    lowres_crop_size = hr_crop_size // scale  # 96//4=24
    lowres_img_shape = tf.shape(lowres_img)[:2]  # (height,width)

    lowres_width = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32
    )
    lowres_height = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32
    )

    highres_width = lowres_width * scale
    highres_height = lowres_height * scale

    lowres_img_cropped = lowres_img[
        lowres_height : lowres_height + lowres_crop_size,
        lowres_width : lowres_width + lowres_crop_size,
    ]  # 24x24
    highres_img_cropped = highres_img[
        highres_height : highres_height + hr_crop_size,
        highres_width : highres_width + hr_crop_size,
    ]  # 96x96

    return lowres_img_cropped, highres_img_cropped

def dataset_object(dataset_cache, training=True):

    ds = dataset_cache
    ds = ds.map(
        lambda lowres, highres: random_crop(lowres, highres, scale=4),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)
    # Batching Data
    ds = ds.batch(16)

    if training:
        # Repeating Data, so that cardinality if dataset becomes infinte
        ds = ds.repeat()
    # prefetching allows later images to be prepared while the current image is being processed
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = dataset_object(train_cache, training=True)
val_ds = dataset_object(val_cache, training=False)

def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value

# initialize the model
print('Computation device: ', device)
# model = srcnn.SRCNN().to(device)
# print(model)

upscale_factor = 4
resblock_layers = 10
channels = 96
kernel = 3

model = edsr.EDSR(upscale_factor,resblock_layers, channels, kernel).to(device)
print(model)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

epochs = 100
print_every = 25
train_loss = 0
batch_num = 0

for epoch_num in range(epochs):
    for x_train, y_train in train_loader:
        image_data = x_train.to(device)
        label = y_train.to(device)

        optimizer.zero_grad()
        pred = model(image_data)
        print(pred.shape, label.shape)
        batch_num += 1
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_num % print_every == 0:
            print('Training Loss: {:.4f}'.format(train_loss / print_every))

    with torch.no_grad():
        val_loss = 0
        model.eval()
        for x_val, y_val in val_loader:
            image_data = x_val.to(device)
            label = y_val.to(device)
            test_pred = model(image_data)
            vloss = criterion(test_pred, label)
            val_loss += vloss.item()

        print('Epoch : {}/{}'.format(epoch_num, epochs))
        print('Training Loss : {:.4f}'.format(train_loss / print_every))
        print('Validation Loss: {:.4f}'.format(val_loss / len(val_loader)))
        train_loss = 0
        model.train()

# # optimizer
# optimizer = optim.Adam(model.parameters(), lr=lr)
# # loss function 
# criterion = nn.MSELoss()

# def psnr(label, outputs, max_val=1.):
#     """
#     Compute Peak Signal to Noise Ratio (the higher the better).
#     PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
#     https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
#     First we need to convert torch tensors to NumPy operable.
#     """
#     label = label.cpu().detach().numpy()
#     outputs = outputs.cpu().detach().numpy()
#     img_diff = outputs - label
#     rmse = math.sqrt(np.mean((img_diff) ** 2))
#     if rmse == 0:
#         return 100
#     else:
#         PSNR = 20 * math.log10(max_val / rmse)
#         return PSNR
    
# def train(model, dataloader):
#     model.train()
#     running_loss = 0.0
#     running_psnr = 0.0
#     for bi, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
#         image_data = data[0].to(device)
#         label = data[1].to(device)
        
#         # zero grad the optimizer
#         optimizer.zero_grad()
#         outputs = model(image_data)
#         loss = criterion(outputs, label)
#         # backpropagation
#         loss.backward()
#         # update the parameters
#         optimizer.step()
#         # add loss of each item (total items in a batch = batch size)
#         running_loss += loss.item()
#         # calculate batch psnr (once every `batch_size` iterations)
#         batch_psnr =  psnr(label, outputs)
#         running_psnr += batch_psnr
#     final_loss = running_loss/len(dataloader.dataset)
#     final_psnr = running_psnr/int(len(train_data)/dataloader.batch_size)
#     return final_loss, final_psnr

# def validate(model, dataloader, epoch):
#     model.eval()
#     running_loss = 0.0
#     running_psnr = 0.0
#     with torch.no_grad():
#         for bi, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
#             image_data = data[0].to(device)
#             label = data[1].to(device)
            
#             outputs = model(image_data)
#             loss = criterion(outputs, label)
#             # add loss of each item (total items in a batch = batch size) 
#             running_loss += loss.item()
#             # calculate batch psnr (once every `batch_size` iterations)
#             batch_psnr = psnr(label, outputs)
#             running_psnr += batch_psnr
#         outputs = outputs.cpu()
#         save_image(outputs, f"outputs/val_sr{epoch}.png")
#     final_loss = running_loss/len(dataloader.dataset)
#     final_psnr = running_psnr/int(len(val_data)/dataloader.batch_size)
#     return final_loss, final_psnr

# train_loss, val_loss = [], []
# train_psnr, val_psnr = [], []
# start = time.time()
# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1} of {epochs}")
#     train_epoch_loss, train_epoch_psnr = train(model, train_loader)
#     val_epoch_loss, val_epoch_psnr = validate(model, val_loader, epoch)
#     print(f"Train PSNR: {train_epoch_psnr:.3f}")
#     print(f"Val PSNR: {val_epoch_psnr:.3f}")
#     train_loss.append(train_epoch_loss)
#     train_psnr.append(train_epoch_psnr)
#     val_loss.append(val_epoch_loss)
#     val_psnr.append(val_epoch_psnr)
# end = time.time()
# print(f"Finished training in: {((end-start)/60):.3f} minutes")

# # loss plots
# plt.figure(figsize=(10, 7))
# plt.plot(train_loss, color='orange', label='train loss')
# plt.plot(val_loss, color='red', label='validataion loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('outputs/loss.png')
# plt.show()
# # psnr plots
# plt.figure(figsize=(10, 7))
# plt.plot(train_psnr, color='green', label='train PSNR dB')
# plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
# plt.xlabel('Epochs')
# plt.ylabel('PSNR (dB)')
# plt.legend()
# plt.savefig('outputs/psnr.png')
# plt.show()
# # save the model to disk
# print('Saving model...')
# torch.save(model.state_dict(), 'outputs/model.pth')
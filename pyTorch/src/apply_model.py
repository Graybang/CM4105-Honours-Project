import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import edsr_stereo_swin
import numpy as np
from functional import to_tensor
from torchvision.utils import save_image
    
# load the super-resolution model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = edsr_stereo_swin.edsr(2,4, 96, 3).to(device)
state_dict = torch.load("pyTorch/src/outputs_stereo/model10.pth")
model.load_state_dict(state_dict)
# set the tile size and overlap
tile_size = 32
overlap = 0

# load the original stereo images
input_image_left = Image.open("0035_L_50.png")
# psnr_input_image_left = Image.open("0035_L.png")
input_image_right = Image.open("0035_R_50.png")

# make sure both images have the same size
w, h = input_image_left.size
input_image_right = input_image_right.resize((w, h))

# calculate the number of tiles needed in each dimension
n_tiles_w = (w - tile_size) // (tile_size - overlap) + 1
n_tiles_h = (h - tile_size) // (tile_size - overlap) + 1

# create empty output stereo images
output_image_left = Image.new("RGB", (w*2, h*2))
output_image_right = Image.new("RGB", (w*2, h*2))

input_image_left.save("pre_left.png")
input_image_right.save("pre_right.png")

# iterate over the tiles and process each one separately
for i in range(n_tiles_w):
    for j in range(n_tiles_h):
        # calculate the position of the tile
        x = i * (tile_size - overlap)
        y = j * (tile_size - overlap)
        
        # extract the tiles from the input stereo images
        tile_left = input_image_left.crop((x, y, x + tile_size, y + tile_size))
        tile_right = input_image_right.crop((x, y, x + tile_size, y + tile_size))

        # convert the tiles to tensors
        tile_left_tensor = to_tensor(tile_left).unsqueeze(0).to(device)
        tile_right_tensor = to_tensor(tile_right).unsqueeze(0).to(device)

        # apply the super-resolution model to the tiles
        output_tensor_left, output_tensor_right = model(tile_left_tensor, tile_right_tensor)

        outputs_L = output_tensor_left.cpu()
        save_image(outputs_L, f"L.png")
        outputs_R = output_tensor_right.cpu()
        save_image(outputs_R, f"R.png")

        # convert the output tensors to images
        # output_image_left_tile = output_tensor_left.squeeze().permute(1, 2, 0).detach().cpu()
        # output_image_left_tile = Image.fromarray(((output_image_left_tile * 255) + 0.5).astype('uint8'), mode='RGB')

        # output_image_right_tile = output_tensor_right.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # output_image_right_tile = Image.fromarray(((output_image_right_tile * 255) + 0.5).astype('uint8'), mode='RGB')
        
        output_image_left_tile = Image.open("L.png")
        output_image_right_tile = Image.open("R.png")

        # paste the output tiles onto the output stereo images
        output_image_left.paste(output_image_left_tile, (2*x, 2*y))
        output_image_right.paste(output_image_right_tile, (2*x, 2*y))

# save the output stereo images
output_image_left.save("left.png")
output_image_right.save("right.png")

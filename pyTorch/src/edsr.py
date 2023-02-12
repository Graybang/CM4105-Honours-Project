# attempted implementation of https://arxiv.org/pdf/1707.02921.pdf

import torch
from torch import nn

class resBlock(nn.Module):
    def __init__(self, channels, kernel):
        super(resBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)
        output = torch.mul(output, 0.1)
        output = torch.add(output, input)

        return output
    
class upsampling(nn.Module):
    def __init__(self, upscale_factor, channels, kernel):
        super(upsampling, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels*upscale_factor**2, kernel, padding=1)
        self.shuffle1 = nn.PixelShuffle(upscale_factor)
        self.conv2 = nn.Conv2d(channels, channels*upscale_factor**2, kernel, padding=1)
        self.shuffle2 = nn.PixelShuffle(upscale_factor)

    def forward(self, output):
        output = self.conv1(output)
        output = self.shuffle1(output)
        output = self.conv2(output)
        output = self.shuffle2(output)

        return output

class EDSR(nn.Module):
    def __init__(self, upscale_factor, layers ,channels ,kernel):
        super(EDSR, self).__init__()

        # First conv2d layer (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(1, channels, kernel, padding=1)

        # Residual blocks
        self.res_block = self.make_res_layer(layers, channels, kernel)

        # Second conv2d layer
        self.conv2 = nn.Conv2d(channels, channels, kernel)

        # Upsampling layers
        self.upscale_block = self.make_ups_layer(upscale_factor, channels, kernel)

        # Third conv2d layer
        self.conv3 = nn.Conv2d(channels, 1, kernel, padding=1)


    def make_res_layer(self, layers, channels, kernel):
        res_blocks = []
        for _ in range(layers):
            res_blocks.append(resBlock(channels, kernel))
        return nn.Sequential(*res_blocks)
    
    def make_ups_layer(self, upscale_factor, channels, kernel, ):
        ups_blocks = []
        for _ in range(upscale_factor):
            ups_blocks.append(upsampling(upscale_factor, channels, kernel))
        return nn.Sequential(*ups_blocks)
    
    def forward(self, output):
        output = self.conv1(output)
        output = self.res_block(output)
        output = self.conv2(output)
        output = self.upscale_block(output)
        output = self.conv3(output)
        return output
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
        self.conv = nn.Conv2d(channels, channels*(upscale_factor**2), kernel, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, output):
        output = self.conv(output)
        output = self.shuffle(output)

        return output

class edsr(nn.Module):
    def __init__(self, upscale_factor, layers ,channels ,kernel):
        super(edsr, self).__init__()

        # First conv2d layer (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(3, channels, kernel, padding=1)

        # Residual blocks
        self.res_block = self.make_res_layer(layers, channels, kernel)

        # Second conv2d layer
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=1)

        # Upsampling layers
        self.upscale_block = self.upsample_layer(1, upscale_factor, channels, kernel)

        # Third conv2d layer
        self.conv3 = nn.Conv2d(channels, 3, kernel, padding=1)


    def make_res_layer(self, layers, channels, kernel):
        res_blocks = []
        for _ in range(layers):
            res_blocks.append(resBlock(channels, kernel))
        return nn.Sequential(*res_blocks)
    
    def upsample_layer(self, layers, upscale_factor, channels, kernel, ):
        upsample_blocks = []
        for _ in range(layers):
            upsample_blocks.append(upsampling(upscale_factor, channels, kernel))
        return nn.Sequential(*upsample_blocks)
    
    def forward(self, output):
        output = self.conv1(output)
        output = self.res_block(output)
        output = self.conv2(output)
        output = self.upscale_block(output)
        output = self.conv3(output)
        return output
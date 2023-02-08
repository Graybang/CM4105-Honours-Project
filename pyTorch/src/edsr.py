# attempted implementation of https://arxiv.org/pdf/1707.02921.pdf

import torch
from torch import nn

class resBlock(nn.Module):
    def __init__(self, x):
        self.conv1 = nn.Conv2d(x)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(x)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(x)
        y = self.conv2(x)
        x = torch.add(x, y)

        return x
    
class upsampling(nn.Module):
    def __init__(self, x):
        self.conv1 = nn.Conv2d(x)
        self.shuffle1 = nn.PixelShuffle(x)
        self.conv2 = nn.Conv2d(x)
        self.shuffle2 = nn.PixelShuffle(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffle1(x)
        x = self.conv2(x)
        x = self.shuffle2(x)

        return x

class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()



    def make_layer(self, block, layers):
        res_blocks = []
        for _ in range(layers):
            res_blocks.append(block())
        return nn.Sequential(*res_blocks)
    
    def forward(self, x):

        

        return x
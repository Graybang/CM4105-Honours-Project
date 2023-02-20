# attempted implementation of https://arxiv.org/pdf/1707.02921.pdf

import torch
from torch import nn
from arch_util import LayerNorm2d

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class resBlock(nn.Module):
    def __init__(self, channels, kernel):
        super(resBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=1)
        self.scam = SCAM(channels)

    def forward(self, input):

        input_L = input[0]
        input_R = input[1]

        output_L = self.conv1(input_L)
        output_R = self.conv1(input_R)

        output_L = self.relu(output_L)
        output_R = self.relu(output_R)

        output_L = self.conv2(output_L)
        output_R = self.conv2(input_R)

        output_L = torch.mul(output_L, 0.1)
        output_L = torch.add(output_L, input_L)

        output_R = torch.mul(output_R, 0.1)
        output_R = torch.add(output_R, input_R)

        output_L, output_R  = self.scam(output_L, output_R)

        return output_L, output_R

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
    
    def forward(self, input_L, input_R):

        output_L = self.conv1(input_L)
        output_R = self.conv1(input_R)

        output = (output_L, output_R)

        output_L, output_R = self.res_block(output)

        output_L = self.conv2(output_L)
        output_R = self.conv2(output_R)

        output_L = self.upscale_block(output_L)
        output_R = self.upscale_block(output_R)

        output_L = self.conv3(output_L)
        output_R = self.conv3(output_R)

        return output_L, output_R
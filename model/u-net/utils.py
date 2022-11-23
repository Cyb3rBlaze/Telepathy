import math

import torch
from torch import nn


# pulled from Dr. Karpathy's minGPT implementation
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# EncoderBlock for spatial dependency extraction       
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        # preserve dimensionality during convolution operations
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding="same")

        self.gelu = GELU()

    def forward(self, x):
        output1 = self.gelu(self.conv1(x))
        output2 = self.gelu(self.conv2(output1))
        output3 = self.gelu(self.conv3(output2))

        return output3

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.blocks = []
        self.blocks += [EncoderBlock(config.in_channels, config.channel_multiplier, config.conv_kernel)]
        for i in range(1, config.num_blocks):
            self.blocks += [EncoderBlock(config.channel_multiplier*i, config.channel_multiplier*2*i, config.conv_kernel)]
        
        self.pool = nn.MaxPool2d(config.pool_kernel, stride=config.pool_stride)

    def forward(self, x):
        # dims[0] = batch size, dims[1] = number of channels, dims[3] + dims[4] = image dimensions
        intermediate_output = self.pool(self.blocks[0](x))
        for i in range(1, len(self.blocks)-1):
            intermediate_output = self.blocks[i](intermediate_output)
            intermediate_output = self.pool(intermediate_output)
        output = self.blocks[-1](intermediate_output)

        return output
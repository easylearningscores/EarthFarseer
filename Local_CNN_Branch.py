import torch
from torch import nn
from modules import *
import torch.nn.functional as F
import numpy as np
import torch.optim as optimizer


class Local_CNN_Branch(nn.Module):
    def __init__(self, in_channels = 2, out_channels = 2):
        super(Local_CNN_Branch, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.upconv = nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # rearrange dimensions to: (B*T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = self.upconv(x)
        # return to original dimensions: (B, T, C, H, W)
        x = x.view(B, T, C, x.shape[2], x.shape[3])
        return x

if __name__ == '__main__':
    x = torch.randn((1, 12, 2, 128, 128))
    y = torch.randn((1, 12, 2, 128, 128))
    model1 = Local_CNN_Branch()
    output = model1(x)
    print("input shape:", x.shape)
    print("output shape:", output.shape)

    def model_memory_usage_in_bytes(model):
        total_bytes = 0
        for param in model.parameters():
            num_elements = np.prod(param.data.shape)
            total_bytes += num_elements * 4  
        return total_bytes
    
    total_bytes = model_memory_usage_in_bytes(model1) 
    mb = total_bytes   / 1048576
    print(f'Total memory used by the model parameters: {mb} MB')
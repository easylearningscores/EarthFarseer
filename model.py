import torch
from torch import nn
from modules import *
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from Global_Fourier_Transformer import GF_Block
from Local_CNN_Branch import *


class Earthfarseer_model(nn.Module):
    def __init__(self, shape_in, hid_S=512, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(Earthfarseer_model, self).__init__()
        T, C, H, W = shape_in
        self.upsampling = Upsampling(in_channels = C, out_channels = C)
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.fourier = GF_Block(img_size=128,
                                patch_size=16,
                                in_channels=2,
                                out_channels=2,
                                input_frames=12,
                                embed_dim=768,
                                depth=12,
                                mlp_ratio=4.,
                                uniform_drop=False,
                                drop_rate=0.,
                                drop_path_rate=0.,
                                norm_layer=None,
                                dropcls=0.)

        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
         #  input shape: torch.Size([1, 12, 2, 32, 32])
        # x_raw = self.upsampling(x_raw)
        B, T, C, H, W = x_raw.shape
        pde_features = self.fourier(x_raw)
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape # BxT, D h w

        z = embed.view(B, T, C_, H_, W_) # B, T, D ,h, w
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y + pde_features




if __name__ == '__main__':
    x = torch.randn((1, 12, 2, 128, 128))
    y = torch.randn((1, 12, 2, 128, 128))
    model1 = Earthfarseer_model(shape_in=(12,2,128,128))
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

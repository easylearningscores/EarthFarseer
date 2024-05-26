import torch
from torch import nn
from modules import *
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from FoTF_module import *
from Temporal_block import *
from utils import *



class Earthfarseer_model(nn.Module):
    def __init__(self, shape_in, hid_S=512, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(Earthfarseer_model, self).__init__()
        T, C, H, W = shape_in
        self.H1 = int(H / 2 ** (N_S / 2)) + 1 if H % 3 == 0 else int(H / 2 ** (N_S / 2))
        self.W1 = int(W / 2 ** (N_S / 2))

        self.fotf_encoder = FoTF(shape_in=shape_in)
        
        self.latent_projection = Encoder(C, hid_S, N_S)
        self.enc = Encoder(C, hid_S, N_S)
        self.TeDev_block = TeDev(T*hid_S, hid_T, N_T, self.H1, self.W1, incep_ker, groups) #
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, input_st_tensors):
        # Spatial block FoTF
        B, T, C, H, W = input_st_tensors.shape
        spatial_feature = self.fotf_encoder(input_st_tensors)

        spatial_feature = spatial_feature.reshape(-1, C, H, W)
        spatial_embed, spatial_skip_feature = self.latent_projection(spatial_feature)
        _, C_, H_, W_ = spatial_embed.shape # BxT, D h w
        spatial_embed = spatial_embed.view(B, T, C_, H_, W_) # B, T, D ,h, w


        # Temporal block TeDev
        spatialtemporal_embed = self.TeDev_block(spatial_embed)
        spatialtemporal_embed = spatialtemporal_embed.reshape(B*T, C_, H_, W_)


        # Decoder
        predictions = self.dec(spatialtemporal_embed, spatial_skip_feature)
        predictions = predictions.reshape(B, T, C, H, W)
        
        return predictions

if __name__ == '__main__':
    x = torch.randn((1, 10, 1, 64, 64))
    y = torch.randn((1, 10, 1, 64, 64))
    model1 = Earthfarseer_model(shape_in=(10, 1, 64, 64))
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

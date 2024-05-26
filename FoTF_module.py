import torch
from torch import nn
from modules import *
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from Global_Fourier_Transformer import GF_Block
from Local_CNN_Branch import *

class FoTF(nn.Module):
    def __init__(self, shape_in, num_interactions=3):
        super(FoTF, self).__init__()
        T, C, H, W = shape_in
        self.lc_block = Local_CNN_Branch(in_channels=C, out_channels=C)
        self.gf_block = GF_Block(
            img_size=H,
            patch_size=16,
            in_channels=C,
            out_channels=C,
            input_frames=T,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            uniform_drop=False,
            drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=None,
            dropcls=0.
        )
        self.up = nn.ConvTranspose2d(C, C, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1)
        self.num_interactions = num_interactions

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        gf_features = self.gf_block(x_raw)
        lc_features = self.lc_block(x_raw)

        for _ in range(self.num_interactions):
            gf_features_up = self.up(gf_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = gf_features_up + lc_features

            gf_features = self.gf_block(combined_features)
            lc_features = self.lc_block(combined_features)

            gf_features_down = self.down(gf_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = gf_features_down + lc_features

            gf_features = self.gf_block(combined_features)
            lc_features = self.lc_block(combined_features)

        return gf_features + lc_features


if __name__ == '__main__':
    shape_in = (12, 2, 128, 128)
    x_raw = torch.randn(1, *shape_in)
    model = FoTF(shape_in)
    finally_output = model(x_raw)
    print(finally_output.shape)


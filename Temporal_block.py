import torch
from torch import nn
from Fourier_computing_unit import *
import math

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class TeDev(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, h, w, incep_ker=[3, 5, 7, 11], groups=8):
        super(TeDev, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(channel_hid)

        self.enc = nn.Sequential(*enc_layers)
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]
        self.h = h
        self.w = w
        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=channel_hid,
            mlp_ratio=4,
            drop=0.,
            drop_path=dpr[i],
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            h = self.h,
            w = self.w)
            for i in range(12)
        ])
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        bias = x
        x = x.reshape(B, T * C, H, W)

        # downsampling
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)
        
        # Spectral Domain
        B, D, H, W = z.shape
        N = H * W
        z = z.permute(0, 2, 3, 1)
        z = z.view(B, N, D)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).permute(0, 2, 1)

        z = z.reshape(B, D, H, W)

        # upsampling
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y + bias

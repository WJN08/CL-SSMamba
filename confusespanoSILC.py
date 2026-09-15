import numpy as np
from typing import Optional
from functools import partial
import transforms
from transforms import GramSchmidtTransform
from torch import Tensor
from mamba_ssm.modules.mamba_simple import Mamba
import random
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
import torch
from torch import nn
import math
from einops import rearrange
from mamba_simple import Mamba
import model_AE_transformer

def split_band(x, move_num, spec_num):
    """
    输入: x (Tensor) - 形状为 [b, c] 的张量
    move_num: 每次移动的步长
    spec_num: 每次切片的大小
    返回: 形状为 [b, n, c] 的张量，其中 n 是根据 move_num 和 spec_num 计算得到的切片数量
    """
    b, c, h, w = x.shape
    slices = []
    for i in range(0, c, move_num):
        if i + spec_num > c:
            slice = x[:, c - spec_num:c, :, :]  # 取最后 spec_num 列
        else:
            slice = x[:, i:i + spec_num, :, :]
        slices.append(slice)
    # 使用 torch.stack 而不是逐个转换到 CPU 再转回 CUDA，提高效率
    slices = torch.stack(slices, dim=1)  # 将切片堆叠成新的维度
    return slices

def Gen_data(matrix, i, j, W):
    matrix = matrix.cpu().detach().numpy()
    rows, cols, bands = matrix.shape
    L = []
    space = int((W - 1) / 2)
    for k in range(i - space, i + space + 1):
        for z in range(j - space, j + space + 1):
            if k < 0 or z < 0 or k >= rows or z >= cols:  # or (k == i and z == j)
                pass
            else:
                L.append(matrix[k, z, :])

    # L = [tensor.cpu().numpy() for tensor in L]  # 将每个张量从 GPU 移动到 CPU 并转换为 NumPy
    L = np.array(L)
    ABS = abs(matrix[i, j, :] - L)
    return np.mean((1 - np.exp(-ABS)), 0)


class Residual_SSMN(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# 等于 PreNorm
class LayerNormalize_SSMN(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class mamba_block1(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Residual_SSMN(LayerNormalize_SSMN(dim, Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=dim,  # Model dimension d_model
                    d_state=64,  # SSM state expansion factor # 64
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    use_fast_path=False,
                )))
            )

    def forward(self, x):
        for attention in self.layers:
            x = attention(x)
        return x


class mamba_block2(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Residual_SSMN(LayerNormalize_SSMN(dim, Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=dim,  # Model dimension d_model
                    d_state=64,  # SSM state expansion factor # 64
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    use_fast_path=False,
                )))
            )

    def forward(self, x):
        for attention in self.layers:
            x = attention(x)
        return x


class mambaLG(nn.Module):
    def __init__(self, num_classes=1, dim=64, depth=1, dropout=0.1, band=30, spec_num=12, spec_rate=0.5, device='0',
                 spa_token=16):
        super(mambaLG, self).__init__()
        self.name = 'mambaLG'
        self.spec_num = spec_num
        self.spec_rate = spec_rate
        self.move_num = int(math.ceil(self.spec_num * self.spec_rate))
        self.device = device
        # dim = band

        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # # 空间分支
        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
            # nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_features2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.conv2d_channel = nn.Sequential(
            nn.Conv2d(dim, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_fusion = nn.Sequential(
            nn.Conv2d(4 * dim, out_channels=dim, kernel_size=(1, 1)),
            # nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

        self.SPAM = mamba_block1(dim, depth)  # 原版

        self.spa_token = spa_token
        self.localSPAM = mamba_block1(dim, depth)

        spe_dim = dim
        self.nn1 = nn.Sequential(
            nn.Linear(dim, spe_dim),
            nn.LayerNorm(spe_dim),
            nn.GELU(),
        )  #
        dim = spe_dim

        # 光谱分支
        num_patch = math.floor((dim - (self.spec_num - self.move_num)) / self.move_num) + \
                    math.ceil(
                        (((dim - (self.spec_num - self.move_num)) % self.move_num) + (self.spec_num - self.move_num)) / \
                        self.move_num)

        self.spe_token1 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.spe_token2 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.SPEM = mamba_block2(self.spec_num, depth)

        self.nn2 = nn.Sequential(
            nn.Linear(self.spec_num * num_patch, dim),  # 原始光谱
            nn.LayerNorm(dim),
            nn.GELU(),
        )  #
        self.outhead = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )

    def forward(self, x, test=False):
        B, H, W, C = x.shape

        x = self.preprocess(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # # 先空间后光谱
        x_spe = x
        # # spatial
        # x = x.permute(0, 3, 1, 2)
        # x1 = self.conv2d_channel(x)
        # x2 = self.conv2d_features1(x)
        # x3 = self.conv2d_features2(x)
        # # x = x1 + x2 + x3 + x
        # x = torch.cat([x, x1, x2, x3], dim=1)
        # x = self.conv2d_fusion(x)
        # x = self.dropout(x)  # .permute(0, 2, 3, 1)
        # x_s1 = x
        # # expand H and W
        # eH = self.spa_token - H % self.spa_token
        # eW = self.spa_token - W % self.spa_token
        # pad = torch.nn.ReflectionPad2d((0, eW, 0, eH)).to(self.device)
        # x = pad(x)
        # bb, cc, hh, ww = x.shape
        # x = rearrange(x, 'b c (nh htoken) (nw wtoken)-> (b nh nw) (wtoken htoken) c', htoken=self.spa_token,
        #               wtoken=self.spa_token)
        # x = self.localSPAM(x)
        # x = rearrange(x, '(b nh nw) (wtoken htoken) c-> b (nh htoken) (nw wtoken) c', htoken=self.spa_token,
        #               wtoken=self.spa_token, nh=hh // self.spa_token, nw=ww // self.spa_token)
        # x = x[:, 0:H, 0:W, :] + x_s1.permute(0, 2, 3, 1)
        #
        # x = rearrange(x, 'b h w c -> b (h w) c')
        # x = self.SPAM(x)
        # x = rearrange(x, 'b (h w) c-> b h w c', h=H, w=W)
        #
        # x_spa = self.nn1(x)

        # # # 光谱
        #print("xsxsxsxs", x_spe.shape)
        #print(x_spe.dtype)
        x_spe = x_spe.view(x_spe.shape[0], -1, x_spe.shape[3])
        ###########################
        data_spa = np.zeros(shape=(100, 100, x_spe.shape[2]))
        for i in range(100):
            for j in range(100):
                data_spa[i, j, :] = Gen_data(x_spe, i, j, 3)
        x_spe = x_spe.cpu().detach().numpy()
        # x_s = np.concatenate((x_spe, data_spa), axis=2)
        x_s = x_spe + data_spa
        #print("xsxsxsxs", x_s.shape)   # 100 100 47
        x_s = torch.tensor(x_s, dtype=torch.float32)
        #############################
        #print("xsxsxsxs", x_s.shape)
        x_s = x_s.unsqueeze(1)
        #print("xsxsxsxs", x_s.shape)
        #print(x_s.dtype)
        #print("xsxsxsxsxsxsxs", x_s.shape)
        # x_s = x_s.reshape(x_s.shape[0], 1, 8, 8, x_s.shape[3]).cuda()
        x_s = x_s.reshape(x_s.shape[0], 1, 10, 10, x_s.shape[3]).cuda()
        #print("xsxsxsxsxsxsxs", x_s.shape)
        #print("x-s", x_s.dtype)
        x_s = self.spe_token1(x_s) + self.spe_token2(x_s) + x_s
        x_s = self.dropout(x_s).squeeze(1).permute(0, 3, 1, 2)
        Patch_pool = torch.nn.AvgPool2d((H, W)).cuda()
        x_s = Patch_pool(x_s)
        #print("xsxsxsxs", x_s.shape)


        x_s = split_band(x_s, self.move_num, self.spec_num)
        bb, nn, cc, hh, ww = x_s.shape
        x_s = rearrange(x_s, 'b n c h w-> (b h w) n c')
        x_s = self.SPEM(x_s)
        x_s = rearrange(x_s, '(b h w) n c-> b (n c) (h w)', h=hh, w=ww).mean(-1)
        x_s = self.nn2(x_s).unsqueeze(1).unsqueeze(1)
        # x = x_spa * x_s  # +x_s2  # + x_s2
        # xout = x
        xout = x_s
        x = rearrange(x, 'b h w c-> b c h w')
        # x = self.outhead(x)
        #print(x.shape)
        if test is True:
            return x, xout, x_s
        else:
            return x, x_s

class mambaLG1(nn.Module):
    def __init__(self, num_classes=1, dim=64, depth=1, dropout=0.1, band=30, spec_num=12, spec_rate=0.5, device='0',
                 spa_token=16):
        super(mambaLG1, self).__init__()
        self.name = 'mambaLG1'
        self.spec_num = spec_num
        self.spec_rate = spec_rate
        self.move_num = int(math.ceil(self.spec_num * self.spec_rate))
        self.device = device
        # dim = band

        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # # 空间分支
        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
            # nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_features2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.conv2d_channel = nn.Sequential(
            nn.Conv2d(dim, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_fusion = nn.Sequential(
            nn.Conv2d(4 * dim, out_channels=dim, kernel_size=(1, 1)),
            # nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

        self.SPAM = mamba_block1(dim, depth)  # 原版

        self.spa_token = spa_token
        self.localSPAM = mamba_block1(dim, depth)

        spe_dim = dim
        self.nn1 = nn.Sequential(
            nn.Linear(dim, spe_dim),
            nn.LayerNorm(spe_dim),
            nn.GELU(),
        )  #
        dim = spe_dim

        # 光谱分支
        num_patch = math.floor((dim - (self.spec_num - self.move_num)) / self.move_num) + \
                    math.ceil(
                        (((dim - (self.spec_num - self.move_num)) % self.move_num) + (self.spec_num - self.move_num)) / \
                        self.move_num)

        self.spe_token1 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.spe_token2 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.SPEM = mamba_block2(self.spec_num, depth)

        self.nn2 = nn.Sequential(
            nn.Linear(self.spec_num * num_patch, dim),  # 原始光谱
            nn.LayerNorm(dim),
            nn.GELU(),
        )  #
        self.outhead = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )

    def forward(self, x, test=False):
        B, H, W, C = x.shape

        x = self.preprocess(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # # 先空间后光谱
        x_spe = x
        # # spatial
        # x = x.permute(0, 3, 1, 2)
        # x1 = self.conv2d_channel(x)
        # x2 = self.conv2d_features1(x)
        # x3 = self.conv2d_features2(x)
        # # x = x1 + x2 + x3 + x
        # x = torch.cat([x, x1, x2, x3], dim=1)
        # x = self.conv2d_fusion(x)
        # x = self.dropout(x)  # .permute(0, 2, 3, 1)
        # x_s1 = x
        # # expand H and W
        # eH = self.spa_token - H % self.spa_token
        # eW = self.spa_token - W % self.spa_token
        # pad = torch.nn.ReflectionPad2d((0, eW, 0, eH)).to(self.device)
        # x = pad(x)
        # bb, cc, hh, ww = x.shape
        # x = rearrange(x, 'b c (nh htoken) (nw wtoken)-> (b nh nw) (wtoken htoken) c', htoken=self.spa_token,
        #               wtoken=self.spa_token)
        # x = self.localSPAM(x)
        # x = rearrange(x, '(b nh nw) (wtoken htoken) c-> b (nh htoken) (nw wtoken) c', htoken=self.spa_token,
        #               wtoken=self.spa_token, nh=hh // self.spa_token, nw=ww // self.spa_token)
        # x = x[:, 0:H, 0:W, :] + x_s1.permute(0, 2, 3, 1)
        #
        # x = rearrange(x, 'b h w c -> b (h w) c')
        # x = self.SPAM(x)
        # x = rearrange(x, 'b (h w) c-> b h w c', h=H, w=W)
        #
        # x_spa = self.nn1(x)

        # # # 光谱
        #print("xsxsxsxs", x_spe.shape)
        #print(x_spe.dtype)
        x_spe = x_spe.view(x_spe.shape[0], -1, x_spe.shape[3])
        ###########################
        data_spa = np.zeros(shape=(100, 100, x_spe.shape[2]))
        for i in range(100):
            for j in range(100):
                data_spa[i, j, :] = Gen_data(x_spe, i, j, 3)
        x_spe = x_spe.cpu().detach().numpy()
        # x_s = np.concatenate((x_spe, data_spa), axis=2)
        x_s = x_spe + data_spa
        #print("xsxsxsxs", x_s.shape)   # 100 100 47
        x_s = torch.tensor(x_s, dtype=torch.float32)
        #############################
        #print("xsxsxsxs", x_s.shape)
        x_s = x_s.unsqueeze(1)
        #print("xsxsxsxs", x_s.shape)
        #print(x_s.dtype)
        #print("xsxsxsxsxsxsxs", x_s.shape)
        # x_s = x_s.reshape(x_s.shape[0], 1, 8, 8, x_s.shape[3]).cuda()
        x_s = x_s.reshape(x_s.shape[0], 1, 10, 10, x_s.shape[3]).cuda()
        #print("xsxsxsxsxsxsxs", x_s.shape)
        #print("x-s", x_s.dtype)
        x_s = self.spe_token1(x_s) + self.spe_token2(x_s) + x_s
        x_s = self.dropout(x_s).squeeze(1).permute(0, 3, 1, 2)
        Patch_pool = torch.nn.AvgPool2d((H, W)).cuda()
        x_s = Patch_pool(x_s)
        #print("xsxsxsxs", x_s.shape)


        x_s = split_band(x_s, self.move_num, self.spec_num)
        bb, nn, cc, hh, ww = x_s.shape
        x_s = rearrange(x_s, 'b n c h w-> (b h w) n c')
        x_s = self.SPEM(x_s)
        x_s = rearrange(x_s, '(b h w) n c-> b (n c) (h w)', h=hh, w=ww).mean(-1)
        x_s = self.nn2(x_s).unsqueeze(1).unsqueeze(1)
        # x = x_spa * x_s  # +x_s2  # + x_s2
        # xout = x
        xout = x_s
        x = rearrange(x, 'b h w c-> b c h w')
        # x = self.outhead(x)
        #print(x.shape)
        if test is True:
            return x, xout, x_s
        else:
            return x, x_s

class mambaLG2(nn.Module):
    def __init__(self, num_classes=1, dim=64, depth=1, dropout=0.1, band=30, spec_num=12, spec_rate=0.5, device='0',
                 spa_token=16):
        super(mambaLG2, self).__init__()
        self.name = 'mambaLG1'
        self.spec_num = spec_num
        self.spec_rate = spec_rate
        self.move_num = int(math.ceil(self.spec_num * self.spec_rate))
        self.device = device
        # dim = band

        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # # 空间分支
        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
            # nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_features2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.conv2d_channel = nn.Sequential(
            nn.Conv2d(dim, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_fusion = nn.Sequential(
            nn.Conv2d(4 * dim, out_channels=dim, kernel_size=(1, 1)),
            # nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

        self.SPAM = mamba_block1(dim, depth)  # 原版

        self.spa_token = spa_token
        self.localSPAM = mamba_block1(dim, depth)

        spe_dim = dim
        self.nn1 = nn.Sequential(
            nn.Linear(dim, spe_dim),
            nn.LayerNorm(spe_dim),
            nn.GELU(),
        )  #
        dim = spe_dim

        # 光谱分支
        num_patch = math.floor((dim - (self.spec_num - self.move_num)) / self.move_num) + \
                    math.ceil(
                        (((dim - (self.spec_num - self.move_num)) % self.move_num) + (self.spec_num - self.move_num)) / \
                        self.move_num)

        self.spe_token1 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.spe_token2 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.SPEM = mamba_block2(self.spec_num, depth)

        self.nn2 = nn.Sequential(
            nn.Linear(self.spec_num * num_patch, dim),  # 原始光谱
            nn.LayerNorm(dim),
            nn.GELU(),
        )  #
        self.outhead = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )

    def forward(self, x, test=False):
        B, H, W, C = x.shape

        x = self.preprocess(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # # 先空间后光谱
        x_spe = x
        # spatial
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv2d_channel(x)
        x2 = self.conv2d_features1(x)
        x3 = self.conv2d_features2(x)
        # x = x1 + x2 + x3 + x
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv2d_fusion(x)
        x = self.dropout(x)  # .permute(0, 2, 3, 1)
        x_s1 = x
        # expand H and W
        eH = self.spa_token - H % self.spa_token
        eW = self.spa_token - W % self.spa_token
        pad = torch.nn.ReflectionPad2d((0, eW, 0, eH)).to(self.device)
        x = pad(x)
        bb, cc, hh, ww = x.shape
        x = rearrange(x, 'b c (nh htoken) (nw wtoken)-> (b nh nw) (wtoken htoken) c', htoken=self.spa_token,
                      wtoken=self.spa_token)
        x = self.localSPAM(x)
        x = rearrange(x, '(b nh nw) (wtoken htoken) c-> b (nh htoken) (nw wtoken) c', htoken=self.spa_token,
                      wtoken=self.spa_token, nh=hh // self.spa_token, nw=ww // self.spa_token)
        x = x[:, 0:H, 0:W, :] + x_s1.permute(0, 2, 3, 1)

        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.SPAM(x)
        x = rearrange(x, 'b (h w) c-> b h w c', h=H, w=W)

        x_spa = self.nn1(x)

        # # # 光谱
        #print("xsxsxsxs", x_spe.shape)
        #print(x_spe.dtype)
        x_spe = x_spe.view(x_spe.shape[0], -1, x_spe.shape[3])
        ###########################
        data_spa = np.zeros(shape=(100, 100, x_spe.shape[2]))
        for i in range(100):
            for j in range(100):
                data_spa[i, j, :] = Gen_data(x_spe, i, j, 3)
        x_spe = x_spe.cpu().detach().numpy()
        # x_s = np.concatenate((x_spe, data_spa), axis=2)
        x_s = x_spe + data_spa
        #print("xsxsxsxs", x_s.shape)   # 100 100 47
        x_s = torch.tensor(x_s, dtype=torch.float32)
        #############################
        #print("xsxsxsxs", x_s.shape)
        x_s = x_s.unsqueeze(1)
        #print("xsxsxsxs", x_s.shape)
        #print(x_s.dtype)
        #print("xsxsxsxsxsxsxs", x_s.shape)
        x_s = x_s.reshape(x_s.shape[0], 1, 10, 10, x_s.shape[3]).cuda()
        #print("xsxsxsxsxsxsxs", x_s.shape)
        #print("x-s", x_s.dtype)
        x_s = self.spe_token1(x_s) + self.spe_token2(x_s) + x_s
        x_s = self.dropout(x_s).squeeze(1).permute(0, 3, 1, 2)
        Patch_pool = torch.nn.AvgPool2d((H, W)).cuda()
        x_s = Patch_pool(x_s)
        #print("xsxsxsxs", x_s.shape)


        x_s = split_band(x_s, self.move_num, self.spec_num)
        bb, nn, cc, hh, ww = x_s.shape
        x_s = rearrange(x_s, 'b n c h w-> (b h w) n c')
        x_s = self.SPEM(x_s)
        x_s = rearrange(x_s, '(b h w) n c-> b (n c) (h w)', h=hh, w=ww).mean(-1)
        x_s = self.nn2(x_s).unsqueeze(1).unsqueeze(1)
        # x = x_spa * x_s  # +x_s2  # + x_s2
        # xout = x
        xout = x_s
        x = rearrange(x, 'b h w c-> b c h w')
        # x = self.outhead(x)
        #print(x.shape)
        if test is True:
            return x, xout, x_s
        else:
            return x, x_s
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: GramSchmidtTransform, input: Tensor):
        #happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)


def RX(hsi_2D):
    m, n = hsi_2D.shape
    X_in = hsi_2D
    X_in_mean = np.mean(X_in, 0)
    X_in = X_in - np.tile(X_in_mean, (m, 1))
    X_in_T = X_in.T
    D = np.cov(X_in_T)
    invD = np.linalg.inv(D)
    out = np.zeros(len(X_in))
    for i in range(len(X_in)):
        x = X_in[i]
        out[i] = np.sqrt(np.dot(np.dot(x.T, invD), x))
    return out, invD

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_layer_dim):
        super(Encoder, self).__init__()
        self.act_f_1 = nn.ReLU()
        self.encoder_layer1 = nn.Linear(input_dim, input_dim // 2, bias=False)
        self.BatchNorm1d_enc_layer1 = nn.BatchNorm1d(input_dim // 2)

        self.encoder_layer2 = nn.Linear(input_dim // 2, input_dim // 4, bias=False)
        self.BatchNorm1d_enc_layer2 = nn.BatchNorm1d(input_dim // 4)

        self.encoder_layer3 = nn.Linear(input_dim // 4, latent_layer_dim, bias=False)
        self.BatchNorm1d_enc_layer3 = nn.BatchNorm1d(latent_layer_dim)

        self.m11 = model_AE_transformer.VisionMamba1(
            patch_size=1,  # patch大小（分块的大小）
            dim=64,  # position embedding的维度
            depth=12,  # 深度
            rms_norm=True,
            image_size=19,  # 图像大小 ABU:10 HAD:8
            residual_in_fp32=True,
            fused_add_norm=True,
            final_pool_type='mean',
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            bimamba_type="v2",
            if_cls_token=True,
            if_devide_out=True,
            use_middle_cls_token=True,
        )
        self.m22 = model_AE_transformer.VisionMamba2(
            patch_size=1,  # patch大小（分块的大小）
            dim=64,  # position embedding的维度
            depth=12,  # 深度
            rms_norm=True,
            image_size=10,  # 图像大小
            residual_in_fp32=True,
            fused_add_norm=True,
            final_pool_type='mean',
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            bimamba_type="v2",
            if_cls_token=True,
            if_devide_out=True,
            use_middle_cls_token=True,
        )
        self.m33 = model_AE_transformer.VisionMamba(
            patch_size=1,  # patch大小（分块的大小）
            dim=64,  # position embedding的维度
            depth=12,  # 深度
            rms_norm=True,
            image_size=10,  # 图像大小
            residual_in_fp32=True,
            fused_add_norm=True,
            final_pool_type='mean',
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            bimamba_type="v2",
            if_cls_token=True,
            if_devide_out=True,
            use_middle_cls_token=True,
        )
        self.m1 = mambaLG(
            num_classes=94,
            dim=94,
            depth=1,
            dropout=0.1,
            band=188,
            spec_num=12,
            spec_rate=0.5,
            device='cuda:0',
            spa_token=16
        )
        self.m2 = mambaLG(
            num_classes=47,
            dim=47,
            depth=1,
            dropout=0.1,
            band=94,
            spec_num=12, #12
            spec_rate=0.5,
            device='cuda:0',
            spa_token=16
        )
        self.m3 = mambaLG(
            num_classes=64,
            dim=64,
            depth=1,
            dropout=0.1,
            band=47,
            spec_num=6, ##6
            spec_rate=0.5,
            device='cuda:0',
            spa_token=16
        )

    def forward(self, x): #100 100 188
        dim0_size = x.shape[0]  # 获取第0维的大小
        dim0_sqrt = torch.sqrt(torch.tensor(dim0_size))
        # 假设 x_sqrt 是一个标量，确保它是一个整数
        x_sqrt = int(dim0_sqrt.item())  # 获取 x_sqrt 的数值
        x_spa = x.reshape(x.shape[1], x.shape[2], x_sqrt, x_sqrt)
        x_spa = x_spa.to(torch.float32)
        #print(x_spa.shape)
        e11 = self.m11(x_spa)  ##100 9400  64 9400
        print("e11", e11.shape)
        dim0 = e11.shape[0]  # 获取第0维的大小
        dim0_s = torch.sqrt(torch.tensor(dim0))
        # 假设 x_sqrt 是一个标量，确保它是一个整数
        x_s = int(dim0_s.item())  # 获取 x_sqrt 的数值
        e11 = e11.view(e11.shape[0], -1, x_s, x_s) # 100 94 10 10
        print("e11", e11.shape)
        #######################################################
        e11 = e11.reshape(e11.shape[0], e11.shape[2], e11.shape[3], e11.shape[1])
        #print("e1", e11.shape)
        e2, spe2 = self.m2(e11)
        #print("e2", e2.shape)  # 1 47 100 100  #100 47 10 10
        # e22 = e22.reshape(1, e22.shape[1], e22.shape[0], -1)
        # e2 = e2 * e22
        e2 = e2.reshape(e2.shape[0], e2.shape[2], e2.shape[3], e2.shape[1])
        enc, spe3 = self.m3(e2)
        # enc1 = enc1.reshape(1, enc1.shape[1], enc1.shape[0], -1)
        # enc = enc1 * enc
        # print("Env",enc.shape) #1 64 100 100
        #print("e1", enc.shape)
        e1 = e11.reshape(e11.shape[0], -1, e11.shape[3])
        e2 = e2.reshape(e2.shape[0], -1, e2.shape[3])  # 100 100 47
        enc = enc.reshape(enc.shape[0], -1, enc.shape[1])
        #print("e1", enc.shape)
        return e1, e2, enc

class Encoder1(nn.Module):
    def __init__(self, input_dim, latent_layer_dim):
        super(Encoder1, self).__init__()
        self.act_f_1 = nn.ReLU()
        self.encoder_layer1 = nn.Linear(input_dim, input_dim // 2, bias=False)
        self.BatchNorm1d_enc_layer1 = nn.BatchNorm1d(input_dim // 2)

        self.encoder_layer2 = nn.Linear(input_dim // 2, input_dim // 4, bias=False)
        self.BatchNorm1d_enc_layer2 = nn.BatchNorm1d(input_dim // 4)

        self.encoder_layer3 = nn.Linear(input_dim // 4, latent_layer_dim, bias=False)
        self.BatchNorm1d_enc_layer3 = nn.BatchNorm1d(latent_layer_dim)

        self.m11 = model_AE_transformer.VisionMamba1(
            patch_size=1,  # patch大小（分块的大小）
            dim=64,  # position embedding的维度
            depth=12,  # 深度
            rms_norm=True,
            image_size=10,  # 图像大小
            residual_in_fp32=True,
            fused_add_norm=True,
            final_pool_type='mean',
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            bimamba_type="v2",
            if_cls_token=True,
            if_devide_out=True,
            use_middle_cls_token=True,
        )
        self.m2 = mambaLG1(
            num_classes=47,
            dim=47,
            depth=1,
            dropout=0.1,
            band=94,
            spec_num=12, #12
            spec_rate=0.5,
            device='cuda:0',
            spa_token=16
        )
        self.m3 = mambaLG1(
            num_classes=64,
            dim=64,
            depth=1,
            dropout=0.1,
            band=47,
            spec_num=6, ##6
            spec_rate=0.5,
            device='cuda:0',
            spa_token=16
        )

    def forward(self, x): #100 100 188
        dim0_size = x.shape[0]  # 获取第0维的大小
        dim0_sqrt = torch.sqrt(torch.tensor(dim0_size))
        # 假设 x_sqrt 是一个标量，确保它是一个整数
        x_sqrt = int(dim0_sqrt.item())  # 获取 x_sqrt 的数值
        x_spa = x.reshape(x.shape[1], x.shape[2], x_sqrt, x_sqrt)
        x_spa = x_spa.to(torch.float32)
        #print(x_spa.shape)
        e11 = self.m11(x_spa)  ##1 94 100 100
        dim0 = e11.shape[0]  # 获取第0维的大小
        dim0_s = torch.sqrt(torch.tensor(dim0))
        # print("e11", e11.shape)
        # 假设 x_sqrt 是一个标量，确保它是一个整数
        x_s = int(dim0_s.item())  # 获取 x_sqrt 的数值
        e11 = e11.reshape(e11.shape[0], -1, x_s, x_s)
        #######################################################
        e11 = e11.reshape(e11.shape[0], e11.shape[2], e11.shape[3], e11.shape[1])
        #print("e1", e11.shape)
        e2, spe2 = self.m2(e11)
        #print("e2", e2.shape)  # 1 47 100 100  #100 47 10 10
        # e22 = e22.reshape(1, e22.shape[1], e22.shape[0], -1)
        # e2 = e2 * e22
        e2 = e2.reshape(e2.shape[0], e2.shape[2], e2.shape[3], e2.shape[1])
        enc, spe3 = self.m3(e2)
        # enc1 = enc1.reshape(1, enc1.shape[1], enc1.shape[0], -1)
        # enc = enc1 * enc
        # print("Env",enc.shape) #1 64 100 100
        #print("e1", enc.shape)
        e1 = e11.reshape(e11.shape[0], -1, e11.shape[3])
        e2 = e2.reshape(e2.shape[0], -1, e2.shape[3])  # 100 100 47
        enc = enc.reshape(enc.shape[0], -1, enc.shape[1])
        #print("e1", enc.shape)
        return e1, e2, enc

class Encoder2(nn.Module):
    def __init__(self, input_dim, latent_layer_dim):
        super(Encoder2, self).__init__()
        self.act_f_1 = nn.ReLU()
        self.encoder_layer1 = nn.Linear(input_dim, input_dim // 2, bias=False)
        self.BatchNorm1d_enc_layer1 = nn.BatchNorm1d(input_dim // 2)

        self.encoder_layer2 = nn.Linear(input_dim // 2, input_dim // 4, bias=False)
        self.BatchNorm1d_enc_layer2 = nn.BatchNorm1d(input_dim // 4)

        self.encoder_layer3 = nn.Linear(input_dim // 4, latent_layer_dim, bias=False)
        self.BatchNorm1d_enc_layer3 = nn.BatchNorm1d(latent_layer_dim)

        self.m11 = model_AE_transformer.VisionMamba1(
            patch_size=1,  # patch大小（分块的大小）
            dim=64,  # position embedding的维度
            depth=12,  # 深度
            rms_norm=True,
            image_size=10,  # 图像大小
            residual_in_fp32=True,
            fused_add_norm=True,
            final_pool_type='mean',
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            bimamba_type="v2",
            if_cls_token=True,
            if_devide_out=True,
            use_middle_cls_token=True,
        )
        self.m2 = mambaLG2(
            num_classes=47,
            dim=47,
            depth=1,
            dropout=0.1,
            band=94,
            spec_num=12, #12
            spec_rate=0.5,
            device='cuda:0',
            spa_token=16
        )
        self.m3 = mambaLG2(
            num_classes=64,
            dim=64,
            depth=1,
            dropout=0.1,
            band=47,
            spec_num=6, ##6
            spec_rate=0.5,
            device='cuda:0',
            spa_token=16
        )

    def forward(self, x): #100 100 188
        dim0_size = x.shape[0]  # 获取第0维的大小
        dim0_sqrt = torch.sqrt(torch.tensor(dim0_size))
        # 假设 x_sqrt 是一个标量，确保它是一个整数
        x_sqrt = int(dim0_sqrt.item())  # 获取 x_sqrt 的数值
        x_spa = x.reshape(x.shape[1], x.shape[2], x_sqrt, x_sqrt)
        x_spa = x_spa.to(torch.float32)
        #print(x_spa.shape)
        e11 = self.m11(x_spa)  ##1 94 100 100
        dim0 = e11.shape[0]  # 获取第0维的大小
        dim0_s = torch.sqrt(torch.tensor(dim0))
        #print("e11", e11.shape)
        # 假设 x_sqrt 是一个标量，确保它是一个整数
        x_s = int(dim0_s.item())  # 获取 x_sqrt 的数值
        e11 = e11.reshape(e11.shape[0], -1, x_s, x_s)
        #######################################################
        e11 = e11.reshape(e11.shape[0], e11.shape[2], e11.shape[3], e11.shape[1])
        #print("e1", e11.shape)
        e2, spe2 = self.m2(e11)
        #print("e2", e2.shape)  # 1 47 100 100  #100 47 10 10
        # e22 = e22.reshape(1, e22.shape[1], e22.shape[0], -1)
        # e2 = e2 * e22
        e2 = e2.reshape(e2.shape[0], e2.shape[2], e2.shape[3], e2.shape[1])
        enc, spe3 = self.m3(e2)
        # enc1 = enc1.reshape(1, enc1.shape[1], enc1.shape[0], -1)
        # enc = enc1 * enc
        # print("Env",enc.shape) #1 64 100 100
        #print("e1", enc.shape)
        e1 = e11.reshape(e11.shape[0], -1, e11.shape[3])
        e2 = e2.reshape(e2.shape[0], -1, e2.shape[3])  # 100 100 47
        enc = enc.reshape(enc.shape[0], -1, enc.shape[1])
        #print("e1", enc.shape)
        return e1, e2, enc

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_layer_dim):
        super(Decoder, self).__init__()
        self.act_f_1 = nn.ReLU()
        self.act_f_2 = nn.Sigmoid()

        self.decoder_layer1 = nn.Linear(latent_layer_dim, input_dim // 4, bias=False)
        self.BatchNorm1d_dec_layer1 = nn.BatchNorm1d(input_dim // 4)

        self.decoder_layer2 = nn.Linear(input_dim // 4 * 2, input_dim // 2, bias=False)
        self.BatchNorm1d_dec_layer2 = nn.BatchNorm1d(input_dim // 2)

        self.decoder_layer3 = nn.Linear(input_dim // 2 * 2, input_dim, bias=False)
        self.BatchNorm1d_dec_layer3 = nn.BatchNorm1d(input_dim)

    def forward(self, enc, e1, e2):
        d1 = self.act_f_1(self.BatchNorm1d_dec_layer1(self.decoder_layer1(enc)))
        #print(d1.shape)
        e1 = e1.reshape(-1, e1.shape[2])
        e2 = e2.reshape(-1, e2.shape[2])
        #print("e1", e1.shape)
        d11 = torch.cat([d1, e2], axis=1)
        #print("d11", d11.shape)
        d2 = self.act_f_1(self.BatchNorm1d_dec_layer2(self.decoder_layer2(d11)))
        #print("d22", d2.shape)
        d22 = torch.cat([d2, e1], axis=1)
        dec = self.act_f_2(self.BatchNorm1d_dec_layer3(self.decoder_layer3(d22)))
        return dec

class Decoder1(nn.Module):
    def __init__(self, input_dim, latent_layer_dim):
        super(Decoder1, self).__init__()
        self.act_f_1 = nn.ReLU()
        self.act_f_2 = nn.Sigmoid()

        self.decoder_layer1 = nn.Linear(latent_layer_dim, input_dim // 4, bias=False)
        self.BatchNorm1d_dec_layer1 = nn.BatchNorm1d(input_dim // 4)

        self.decoder_layer2 = nn.Linear(input_dim // 4 * 2, input_dim // 2, bias=False)
        self.BatchNorm1d_dec_layer2 = nn.BatchNorm1d(input_dim // 2)

        self.decoder_layer3 = nn.Linear(input_dim // 2 * 2, input_dim, bias=False)
        self.BatchNorm1d_dec_layer3 = nn.BatchNorm1d(input_dim)

    def forward(self, enc, e1, e2):
        e1 = e1.reshape(-1, e1.shape[2])
        e2 = e2.reshape(-1, e2.shape[2])
        d1 = self.act_f_1(self.BatchNorm1d_dec_layer1(self.decoder_layer1(enc)))
        d11 = torch.cat([d1, e2], axis=1)
        d2 = self.act_f_1(self.BatchNorm1d_dec_layer2(self.decoder_layer2(d11)))
        d22 = torch.cat([d2, e1], axis=1)
        dec = self.act_f_2(self.BatchNorm1d_dec_layer3(self.decoder_layer3(d22)))
        return dec

class Decoder2(nn.Module):
    def __init__(self, input_dim, latent_layer_dim):
        super(Decoder2, self).__init__()
        self.act_f_1 = nn.ReLU()
        self.act_f_2 = nn.Sigmoid()

        self.decoder_layer1 = nn.Linear(latent_layer_dim, input_dim // 4, bias=False)
        self.BatchNorm1d_dec_layer1 = nn.BatchNorm1d(input_dim // 4)

        self.decoder_layer2 = nn.Linear(input_dim // 4 * 2, input_dim // 2, bias=False)
        self.BatchNorm1d_dec_layer2 = nn.BatchNorm1d(input_dim // 2)

        self.decoder_layer3 = nn.Linear(input_dim // 2 * 2, input_dim, bias=False)
        self.BatchNorm1d_dec_layer3 = nn.BatchNorm1d(input_dim)

    def forward(self, enc, e1, e2):
        d1 = self.act_f_1(self.BatchNorm1d_dec_layer1(self.decoder_layer1(enc)))
        print(d1.shape)
        e1 = e1.reshape(-1, e1.shape[2])
        e2 = e2.reshape(-1, e2.shape[2])
        #print("e1", e1.shape)
        d11 = torch.cat([d1, e2], axis=1)
        print("d11", d11.shape)
        d2 = self.act_f_1(self.BatchNorm1d_dec_layer2(self.decoder_layer2(d11)))
        #print("d22", d2.shape)
        d22 = torch.cat([d2, e1], axis=1)
        dec = self.act_f_2(self.BatchNorm1d_dec_layer3(self.decoder_layer3(d22)))
        return dec

class OrthoAE_Unet(nn.Module):
    def __init__(self, input_dim, latent_layer_dim, args):
        super(OrthoAE_Unet, self).__init__()
        self.encoder = Encoder(input_dim, latent_layer_dim)
        self.encoder1 = Encoder1(input_dim, latent_layer_dim)
        # self.encoder2 = Encoder2(input_dim, latent_layer_dim)
        self.decoder = Decoder(input_dim, latent_layer_dim)
        self.decoder1 = Decoder1(input_dim, latent_layer_dim)
        # self.decoder2 = Decoder2(input_dim, latent_layer_dim)

        self._excitation = nn.Sequential(
            nn.Linear(in_features=latent_layer_dim, out_features=round(latent_layer_dim // 2), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(latent_layer_dim // 2), out_features=latent_layer_dim, bias=False),
            nn.Sigmoid(),
        )

        self.OrthoAttention = Attention()
        self.F_C_A = transforms.GramSchmidtTransform.build(latent_layer_dim, 100)

    def forward(self, x, args):
        ########1###################
        e1, e2, enc = self.encoder(x)
        #print("enc", e1.shape)
        fea_1 = self.OrthoAttention(self.F_C_A, enc)
        # print(fea_1.shape)
        fea_1 = fea_1.reshape(fea_1.shape[1], fea_1.shape[0])
        b, c = fea_1.size(0), fea_1.size(1)
        attention_4d = self._excitation(fea_1).view(b, c, 1, 1)
        attention_2d = attention_4d.view(-1, c)
        attention_fea = enc * attention_2d
        #print(attention_fea.shape)
        attention_fea = attention_fea.reshape(-1, attention_fea.shape[2])
        dec = self.decoder(attention_fea, e1, e2)
        ###################2######################
        e11, e22, encc = self.encoder1(x)
        fea_11 = self.OrthoAttention(self.F_C_A, encc)
        fea_11 = fea_11.reshape(fea_11.shape[1], fea_11.shape[0])
        bb, cc = fea_11.size(0), fea_11.size(1)
        attention_4dd = self._excitation(fea_11).view(bb, cc, 1, 1)
        attention_2dd = attention_4dd.view(-1, cc)
        attention_feaa = encc * attention_2dd
        attention_feaa = attention_feaa.reshape(-1, attention_feaa.shape[2])
        dec1 = self.decoder1(attention_feaa, e11, e22)
        #####################3#########################
        # e111, e222, enccc = self.encoder2(x)
        # fea_111 = self.OrthoAttention(self.F_C_A, enccc)
        # fea_111 = fea_111.reshape(fea_111.shape[1], fea_111.shape[0])
        # bbb, ccc = fea_111.size(0), fea_111.size(1)
        # attention_4ddd = self._excitation(fea_111).view(bbb, ccc, 1, 1)
        # attention_2ddd = attention_4ddd.view(-1, ccc)
        # attention_feaaa = encc * attention_2ddd
        # attention_feaaa = attention_feaaa.reshape(-1, attention_feaaa.shape[2])
        # dec2 = self.decoder2(attention_feaaa, e111, e222)

        # 计算 dec 和 dec1 的平均值
        # combined_dec = (dec + dec1 + dec2) / 3
        combined_dec = (dec + dec1 ) / 2

        # Min-Max 归一化到 [0, 1] 范围
        dec_min = combined_dec.min()
        dec_max = combined_dec.max()
        dec_normalized = (combined_dec - dec_min) / (dec_max - dec_min)

        return attention_fea, dec_normalized, dec, dec1, _

import torch
from torch import nn
import numpy as np
from torch import nn, optim

import transforms
from transforms import GramSchmidtTransform
from torch import Tensor


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

    def forward(self, x):
        x = x.to(torch.float32)
        print(x.shape)
        e1 = self.act_f_1(self.BatchNorm1d_enc_layer1(self.encoder_layer1(x)))
        print(e1.shape)
        e2 = self.act_f_1(self.BatchNorm1d_enc_layer2(self.encoder_layer2(e1)))
        print(e2.shape)
        enc = self.act_f_1(self.BatchNorm1d_enc_layer3(self.encoder_layer3(e2)))
        print(enc.shape)
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
        print(enc.shape)
        d1 = self.act_f_1(self.BatchNorm1d_dec_layer1(self.decoder_layer1(enc)))
        print(d1.shape)
        d11 = torch.cat([d1, e2], axis=1)
        print(d11.shape)
        d2 = self.act_f_1(self.BatchNorm1d_dec_layer2(self.decoder_layer2(d11)))
        print(d2.shape)
        d22 = torch.cat([d2, e1], axis=1)
        print(d22.shape)
        dec = self.act_f_2(self.BatchNorm1d_dec_layer3(self.decoder_layer3(d22)))
        print(dec.shape)
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
        d1 = self.act_f_1(self.BatchNorm1d_dec_layer1(self.decoder_layer1(enc)))
        d11 = torch.cat([d1, e2], axis=1)
        d2 = self.act_f_1(self.BatchNorm1d_dec_layer2(self.decoder_layer2(d11)))
        d22 = torch.cat([d2, e1], axis=1)
        dec = self.act_f_2(self.BatchNorm1d_dec_layer3(self.decoder_layer3(d22)))
        return dec

class Decoder2(nn.Module):
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
        d11 = torch.cat([d1, e2], axis=1)
        d2 = self.act_f_1(self.BatchNorm1d_dec_layer2(self.decoder_layer2(d11)))
        d22 = torch.cat([d2, e1], axis=1)
        dec = self.act_f_2(self.BatchNorm1d_dec_layer3(self.decoder_layer3(d22)))
        return dec

class OrthoAE_Unet(nn.Module):
    def __init__(self, input_dim, latent_layer_dim, args):
        super(OrthoAE_Unet, self).__init__()
        self.encoder = Encoder(input_dim, latent_layer_dim)
        self.decoder = Decoder(input_dim, latent_layer_dim)
        self.decoder1 = Decoder1(input_dim, latent_layer_dim)

        self._excitation = nn.Sequential(
            nn.Linear(in_features=latent_layer_dim, out_features=round(latent_layer_dim // 2), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(latent_layer_dim // 2), out_features=latent_layer_dim, bias=False),
            nn.Sigmoid(),
        )

        self.OrthoAttention = Attention()
        self.F_C_A = transforms.GramSchmidtTransform.build(latent_layer_dim, 100)

    def forward(self, x, args):

        e1, e2, enc = self.encoder(x)
        # print(x.shape)
        dim0_size = x.shape[0]  # 获取第0维的大小
        dim0_sqrt = torch.sqrt(torch.tensor(dim0_size))
        # 假设 x_sqrt 是一个标量，确保它是一个整数
        x_sqrt = int(dim0_sqrt.item())  # 获取 x_sqrt 的数值

        d3_enc_fea = enc.unsqueeze(0).reshape((x_sqrt, x_sqrt, enc.shape[1])).permute(2, 0, 1).unsqueeze(0)
        fea_1 = self.OrthoAttention(self.F_C_A, d3_enc_fea)  # 1 64
        b, c = fea_1.size(0), fea_1.size(1)
        attention_4d = self._excitation(fea_1).view(b, c, 1, 1)
        attention_2d = attention_4d.view(-1, c)
        attention_fea = enc * attention_2d

        dec = self.decoder(attention_fea, e1, e2)
        # dec1 = self.decoder1(attention_fea, e1, e2)
        #
        # # 计算 dec 和 dec1 的平均值
        # combined_dec = (dec + dec1) / 2

        # Min-Max 归一化到 [0, 1] 范围
        # dec_min = combined_dec.min()
        # dec_max = combined_dec.max()
        # dec_normalized = (combined_dec - dec_min) / (dec_max - dec_min)
        dec_normalized = dec
        return attention_fea, dec_normalized

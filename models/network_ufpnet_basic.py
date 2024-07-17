import torch
import torch.nn as nn
import torch.fft

from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import scipy.io as sio
from scipy.io import loadmat
import os.path

import scipy.misc
from scipy import ndimage
import time

import kornia

# for pytorch version >= 1.9.0

"""
# --------------------------------------------
main code of MDFPNet
# Baopeng Li (github: https://github.com/BP113)
# 03/Mar/2023
# --------------------------------------------
"""

"""
# --------------------------------------------
# (1) Prior module; 
# --------------------------------------------
"""

import torch.nn.functional as F
class resnet0(nn.Module):
    def __init__(self, channels):
        super(resnet0, self).__init__()
        self.channels = channels
        self.resx1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),      # check dilation=0 results
                                  # nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  )
        self.resx2 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  )
    def forward(self, input):
        x1 = F.relu(input + self.resx1(input))
        x2 = F.relu(x1 + self.resx2(x1))
        return x2

class resnet_last(nn.Module):
    def __init__(self, channels):
        super(resnet_last, self).__init__()
        self.channels = channels
        self.resx1 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            )
        self.resx2 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            )
        self.resx3 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            )
        self.resx4 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            )
        self.resx5 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(self.channels),
            )

    def forward(self, input):
        x1 = F.relu(input + self.resx1(input))
        x2 = F.relu(x1 + self.resx2(x1))
        x3 = F.relu(x2 + self.resx3(x2))
        x4 = F.relu(x3 + self.resx4(x3))
        x5 = F.relu(x4 + self.resx5(x4))
        return x5

class resnet(nn.Module):
    def __init__(self, channels):
        super(resnet, self).__init__()
        self.channels = channels
        self.resx1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),      # check dilation=0 results
                                  # nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  )
        self.resx2 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  )
        self.resx3 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels,  kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  )
        self.resx4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  )
        self.resx5 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  # nn.BatchNorm2d(self.channels),
                                  )
    def forward(self, input):
        x1 = F.relu(input + self.resx1(input))
        x2 = F.relu(x1 + self.resx2(x1))
        x3 = F.relu(x2 + self.resx3(x2))
        x4 = F.relu(x3 + self.resx4(x3))
        x5 = F.relu(x4 + self.resx5(x4))
        return x5


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# --------------------------------------------
"""
class DataNet0(nn.Module):
    def __init__(self):
        super(DataNet0, self).__init__()
        self.n1 = 256
        self.n2 = 256

    # --------------------------------------------
    # Input: xx: original signal (HR spectrum)
    # Output: yy_c: sampling output (without abs)
    # --------------------------------------------
    def A_LinearOperator(self, xx, Masks, n1_LR, n2_LR, CTF):
        L = Masks.shape[0]
        n_batch = xx.shape[0]
        xx_c = torch.zeros((n_batch, L, n1_LR, n2_LR), dtype=torch.cfloat).cuda()
        for k in range(0, L):    # L-1
            index_x = Masks[k, 0]
            index_y = Masks[k, 1]
            xx_c[:, k, :, :] = xx[:, 0, index_x - 1 : index_x + n1_LR - 1, index_y - 1 : index_y + n2_LR - 1] * CTF       # torch.mul(a,b)  ==   a*b
        yy_c = torch.fft.ifft2(torch.fft.ifftshift(xx_c, dim=(-2, -1)))
        return yy_c

    # --------------------------------------------
    # Input: yy_c: sampling output (without abs)
    # Output: xx_mean_large: original signal (HR spectrum)
    # --------------------------------------------
    def A_Inverse_LinearOperator(self, yy_c, Masks, n1, n2, CTF):
        [n_batch, L, n1_LR, n2_LR] = yy_c.size()
        xx_c = torch.fft.fftshift(torch.fft.fft2(yy_c), dim=(-2, -1))
        xx = torch.zeros((n_batch, L, max(Masks[:, 0]) - min(Masks[:, 0]) + n1_LR, max(Masks[:, 1]) - min(Masks[:, 1]) + n2_LR), dtype=torch.cfloat).cuda()
        for k in range(0, L):
            index_x = Masks[k, 0]
            index_y = Masks[k, 1]
            xx[:, k, index_x-min(Masks[:, 0]) : index_x-min(Masks[:, 0])+n1_LR, index_y-min(Masks[:, 1]) : index_y-min(Masks[:, 1])+n2_LR] = xx_c[:, k, :, :] * torch.conj(CTF)
        xx_mean = xx.mean(axis=1)
        xx_mean_large = torch.zeros((n_batch, n1, n2), dtype=torch.cfloat).cuda()
        xx_mean_large[:, Masks[:, 0].min()-1 : Masks[:, 0].min()+xx_mean.shape[1]-1, Masks[:, 1].min()-1 : Masks[:, 1].min()+xx_mean.shape[2]-1] = xx_mean[:,:,:]
        xx_mean_large = torch.unsqueeze(xx_mean_large, 1)
        return xx_mean_large

    def FP_basic(self, Img_rnew, Y, n1, n2, Masks, CTF, T):
        [n_batch, L, n1_LR, n2_LR] = Y.size()
        Object_FT = torch.fft.fftshift(torch.fft.fft2(Img_rnew), dim=(-2, -1))  # Frequency spectrum
        for t in range(0, T):
            for k in range(0, L):  # L-1
                index_x = Masks[k, 0]
                index_y = Masks[k, 1]
                Object_FT1 = Object_FT[:, 0, index_x - 1: index_x + n1_LR - 1, index_y - 1: index_y + n2_LR - 1]
                Img_low_FT = Object_FT1 * CTF
                Img_low = torch.fft.ifft2(torch.fft.ifftshift(Img_low_FT, dim=(-2, -1)))
                Img_low_new = torch.sqrt(Y[:, k, :, :]) * torch.exp(1j * torch.angle(Img_low))
                Object_part_FT = torch.fft.fftshift(torch.fft.fft2(Img_low_new), dim=(-2, -1)) * CTF
                Object_FT[:, 0, index_x - 1: index_x + n1_LR - 1, index_y - 1: index_y + n2_LR - 1] = (1 - CTF) * Object_FT1 + Object_part_FT

        im_rc = torch.fft.ifft2(torch.fft.ifftshift(Object_FT, dim=(-2, -1)))
        im_ra = torch.abs(im_rc)
        im_rp = torch.angle(im_rc)

        return im_ra, im_rp, im_rc

    def forward(self, Img_c, Y, n1, n2, Masks, CTF, T):    # Y: measurement images,
        [Img_ra, Img_rp, Img_rc] = self.FP_basic(Img_c, Y, n1, n2, Masks, CTF, T)
        Img_ra = torch.clamp(Img_ra, 0., 100.)  # positivity
        return Img_ra, Img_rp, Img_rc

class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()
        self.n1 = 256
        self.n2 = 256
        self.channels = 1
        self.eps = 1e-6
        self.lamb = torch.nn.Parameter(torch.Tensor([0.1/5]).cuda(), requires_grad=True)       # 0.5
        self.eta1 = torch.nn.Parameter(torch.Tensor([0.1]).cuda(), requires_grad=True)  # original 0.1

    # --------------------------------------------
    # Input: xx: original signal (HR spectrum)
    # Output: yy_c: sampling output (without abs) 
    # --------------------------------------------
    def A_LinearOperator(self, xx, Masks, n1_LR, n2_LR, CTF):
        L = Masks.shape[0]
        n_batch = xx.shape[0]
        xx_c = torch.zeros((n_batch, L, n1_LR, n2_LR), dtype=torch.cfloat).cuda()
        for k in range(0, L):
            index_x = Masks[k, 0]
            index_y = Masks[k, 1]
            xx_c[:, k, :, :] = xx[:, 0, index_x - 1 : index_x + n1_LR - 1, index_y - 1 : index_y + n2_LR - 1] * CTF       # torch.mul(a,b)  ==   a*b
        yy_c = torch.fft.ifft2(torch.fft.ifftshift(xx_c, dim=(-2, -1)))
        return yy_c

    # --------------------------------------------
    # Input: yy_c: sampling output (without abs) 
    # Output: xx_mean_large: original signal (HR spectrum)
    # --------------------------------------------
    def A_Inverse_LinearOperator(self, yy_c, Masks, n1, n2, CTF):
        [n_batch, L, n1_LR, n2_LR] = yy_c.size()
        xx_c = torch.fft.fftshift(torch.fft.fft2(yy_c), dim=(-2, -1))
        xx = torch.zeros((n_batch, L, max(Masks[:, 0]) - min(Masks[:, 0]) + n1_LR, max(Masks[:, 1]) - min(Masks[:, 1]) + n2_LR), dtype=torch.cfloat).cuda()
        for k in range(0, L):
            index_x = Masks[k, 0]
            index_y = Masks[k, 1]
            xx[:, k, index_x-min(Masks[:, 0]) : index_x-min(Masks[:, 0])+n1_LR, index_y-min(Masks[:, 1]) : index_y-min(Masks[:, 1])+n2_LR] = xx_c[:, k, :, :] * torch.conj(CTF)
        xx_mean = xx.mean(axis=1)
        xx_mean_large = torch.zeros((n_batch, n1, n2), dtype=torch.cfloat).cuda()
        xx_mean_large[:, Masks[:, 0].min()-1 : Masks[:, 0].min()+xx_mean.shape[1]-1, Masks[:, 1].min()-1 : Masks[:, 1].min()+xx_mean.shape[2]-1] = xx_mean[:,:,:]
        xx_mean_large = torch.unsqueeze(xx_mean_large, 1)
        return xx_mean_large

    def AFFP_0(self, Img_ra, Img_rnew, Y, n1, n2, Masks, CTF, T):         # initialization step, get the phase results.  the phase is zero at first.
        [n_batch, L, n1_LR, n2_LR] = Y.size()
        z_f = torch.fft.fftshift(torch.fft.fft2(Img_rnew), dim=(-2, -1))   # Frequency spectrum
        Bz = self.A_LinearOperator(z_f, Masks, n1_LR, n2_LR, CTF)
        Cz_update = Bz - torch.sqrt(Y) * torch.exp(1j * torch.angle(Bz))
        z_f_update = self.A_Inverse_LinearOperator(Cz_update, Masks, n1, n2, CTF)
        temp_x = Img_ra * torch.fft.ifft2(torch.fft.ifftshift(z_f, dim=(-2, -1))) / (torch.abs(torch.fft.ifft2(torch.fft.ifftshift(z_f, dim=(-2, -1))))+self.eps)
        z_f_update1 = z_f - torch.fft.fftshift(torch.fft.fft2(temp_x), dim=(-2, -1))
        z_f_new = z_f - 10 * self.eta1 * z_f_update - 10 * self.eta1 * 10 * self.lamb * z_f_update1
        im_rc = torch.fft.ifft2(torch.fft.ifftshift(z_f_new, dim=(-2, -1)))
        im_ra = torch.abs(im_rc)
        return im_ra, im_rc

    def forward(self, Img_a, Img_c, Y, n1, n2, Masks, CTF, T):    # Y: measurement images,
        [Img_ra, Img_rc] = self.AFFP_0(Img_a, Img_c, Y, n1, n2, Masks, CTF, T)
        # Img_deta = torch.clamp(Img_deta, 0., 100.)  # positivity
        return Img_ra, Img_rc

"""
# --------------------------------------------
# main MDFPNet
# deep unfolding Fourier ptychography network
# --------------------------------------------
"""
import heapq

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)

class UFPNet(nn.Module):
    def __init__(self, n_iter=8, batch_size=4, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):     # delete h_nc=64,
        super(UFPNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d0 = DataNet0().to(self.device)
        self.d = DataNet().to(self.device)
        self.num_Z = 31
        # filter for initializing Z
        self.convZ = nn.Conv2d(1, self.num_Z, kernel_size=3, stride = 1, padding= 1,  dilation = 1)

        # filter for initializing B and Z
        self.C_z_const = filter.expand(self.num_Z, 1, -1, -1).clone()
        self.C_z = nn.Parameter(self.C_z_const, requires_grad=True)

        self.S = n_iter
        self.batch_size = batch_size
        self.T0 = 3  # test
        self.T = 1  # number of iteration
        self.n1_LR = 50
        self.n2_LR = 50
        self.n1 = 256
        self.n2 = 256
        self.CTF = torch.from_numpy(loadmat(os.path.join('trainsets/data', 'pupil.mat'))['pupil'].astype('float32')).to(self.device)
        self.Masks = torch.from_numpy(loadmat(os.path.join('trainsets/data', 'Masks.mat'))['Masks'].astype('int16')).to(self.device)
        # initial resnet
        self.proxNet_S = self.make_resnet(self.S, self.num_Z+1)
        # fine-tune at the last layer
        self.proxNnet_last = resnet_last(self.num_Z+1)
        self.eta2const = 1
        self.eta2 = torch.Tensor([self.eta2const])                       # initialization for eta1  at all stages
        self.eta2S = self.make_coeff(self.S, self.eta2)                  # learnable in iterative process

    def make_coeff(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1).clone() # test error, so adding this one
        coeff = nn.Parameter(data=const_f, requires_grad = True)
        return coeff

    def make_resnet(self, iters, channel):
        layers = []
        for i in range(iters):
            layers.append(resnet(channel))
        return nn.Sequential(*layers)

    def forward(self, y):
        '''
        y: tensor, NxCxWxH
        '''
        # save mid-updating results
        ListImg = []
        # initialization & pre-calculation $ 0 iteration
        lowersMean = torch.mean(y, dim=1, keepdim=True)
        Img_r00 = kornia.geometry.rescale(lowersMean, (5.12, 5.12), antialias=True)
        Img_r00 = Img_r00 / torch.max(Img_r00)
        Img_rec, Img_rp0, Img_u_rnew = self.d0(Img_r00, y, self.n1, self.n2, self.Masks, self.CTF, self.T0)
        Z = F.relu(self.convZ(Img_rec))
        # unfolding Fourier ptychography
        for i in range(self.S):
            Img_u_ra, Img_u_rc = self.d(Img_rec, Img_u_rnew, y, self.n1, self.n2, self.Masks, self.CTF, self.T)
            Img_next = (1 - self.eta2S[i]) * Img_rec + self.eta2S[i] * Img_u_ra
            input_dual = torch.cat((Img_next, Z), dim=1)
            out_dual = self.proxNet_S[i](input_dual)
            Img_rec = out_dual[:, :1, :, :]
            Z = out_dual[:, 1:, :, :]
            ListImg.append(Img_rec)
            Img_u_rnew = Img_u_rc

        out_dual = self.proxNnet_last(out_dual)  # fine-tune
        Img_rec = out_dual[:, :1, :, :]
        ListImg.append(Img_rec)

        return ListImg


if __name__ == '__main__':

    x = torch.rand(1, 3, 96, 96)
    print(x.size())

    x1 = torch.rand(4, 6, 100, 100)
    print(x1.size())

    torch.max(x1)
    print(torch.max(x1))
    print(torch.min(x1))


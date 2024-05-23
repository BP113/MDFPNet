import random

import numpy as np
import torch
import torch.utils.data as data

import utils.utils_fp
import utils.utils_image as util
from utils import utils_deblur
from utils import utils_sisr
import os

from scipy import ndimage
from scipy.io import loadmat
# import hdf5storage
import matplotlib.pyplot as plt # plt 用于显示图片


class DatasetUFPNet(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/k/sf/sigma for USRNet.
    # Only "paths_H" and kernel is needed, synthesize L on-the-fly.
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetUFPNet, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 1
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 256
        self.n1_LR = 50
        self.n2_LR = 50
        self.sigma_max = self.opt['sigma_max'] if self.opt['sigma_max'] is not None else 5
        self.sf_validation = opt['sf_validation'] if opt['sf_validation'] is not None else 3

        # -------------------
        # get the path of H
        # -------------------
        # self.paths_Ha = util.get_image_paths(opt['dataroot_H'])  # return None if input is None
        self.paths_Ha = util.get_image_paths(opt['dataroot_Ha'])
        self.paths_Hp = util.get_image_paths(opt['dataroot_Hp'])
        self.count = 0
        self.pupil = loadmat(os.path.join('trainsets/data', 'pupil.mat'))['pupil'].astype('float32')   #.to(self.device)
        self.Masks = loadmat(os.path.join('trainsets/data', 'Masks.mat'))['Masks'].astype('int16')

    def __getitem__(self, index):

        # -------------------
        # get H image
        # -------------------
        Ha_path = self.paths_Ha[index]
        img_Ha = util.imread_uint(Ha_path, self.n_channels)

        Hp_path = self.paths_Hp[index]
        img_Hp = util.imread_uint(Hp_path, self.n_channels)

        L_path = Ha_path

        if self.opt['phase'] == 'train':

            self.count += 1

            # ----------------------------
            # randomly crop the patch
            # ----------------------------
            H, W, _ = img_Ha.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_Ha = img_Ha[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            H, W, _ = img_Hp.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_Hp = img_Hp[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            mode = np.random.randint(0, 8)
            patch_Ha = util.augment_img(patch_Ha, mode=mode)
            patch_Hp = util.augment_img(patch_Hp, mode=mode)

            patch_Ha1 = util.uint2single(patch_Ha)
            patch_Hp1 = util.uint2single(patch_Hp)

            if np.random.randint(0, 8) == 1:
                noise_level = 0/255.0
            else:
                noise_level = np.random.randint(0, self.sigma_max)/255.0

            patch_Ha1 = patch_Ha1 + np.random.normal(0, noise_level, patch_Ha1.shape)
            patch_Hp1 = patch_Hp1 + np.random.normal(0, noise_level, patch_Hp1.shape)

            patch_Hc = patch_Ha1 * np.exp(1j*patch_Hp1)

            xx_c = np.fft.fft2(patch_Hc, axes=(0, 1))
            xx_c = np.fft.fftshift(xx_c, axes=0)
            X = np.fft.fftshift(xx_c, axes=1)

            # ---------------------------
            # Low-quality FP image
            # ---------------------------
            Y = utils.utils_fp.A_LinearOperator(X, self.Masks, self.pupil, self.n1_LR, self.n2_LR)
            img_L = np.power(np.abs(Y), 2)


            # ---------------------------
            # noise level
            # ---------------------------
            if np.random.randint(0, 8) == 1:
                noise_level = 0/255.0
            else:
                noise_level = np.random.randint(0, self.sigma_max)/(255.0*100)
                # noise_level = 0

            # add Gaussian noise
            img_L = img_L + np.abs(np.random.normal(0, noise_level, img_L.shape))

            img_Ha = patch_Ha
            img_Hp = patch_Hp


        else:
            # -------------------
            # get H image
            # -------------------
            Ha_path = self.paths_Ha[index]
            img_Ha = util.imread_uint(Ha_path, self.n_channels)
            Hp_path = self.paths_Hp[index]
            img_Hp = util.imread_uint(Hp_path, self.n_channels)
            L_path = Ha_path

            img_Ha1 = util.uint2single(img_Ha)
            img_Hp1 = util.uint2single(img_Hp)

            img_Hc = img_Ha1 * np.exp(1j*img_Hp1)

            xx_c = np.fft.fft2(img_Hc, axes=(0, 1))
            xx_c = np.fft.fftshift(xx_c, axes=0)
            X = np.fft.fftshift(xx_c, axes=1)

            Y = utils.utils_fp.A_LinearOperator(X, self.Masks, self.pupil, self.n1_LR, self.n2_LR)
            img_L = np.power(np.abs(Y), 2)

            noise_level = 0./255.0  # validation noise level
            # add noise
            # img_L = img_L + np.random.normal(0, noise_level, img_L.shape)

        img_Ha, img_Hp, img_L = util.uint2tensor3(img_Ha), util.uint2tensor3(img_Hp), util.single2tensor3(img_L)
        noise_level = torch.FloatTensor([noise_level]).view([1,1,1])

        return {'L': img_L, 'Ha': img_Ha, 'Hp': img_Hp, 'sigma': noise_level, 'L_path': L_path, 'Ha_path': Ha_path}

    def __len__(self):
        return len(self.paths_Ha)

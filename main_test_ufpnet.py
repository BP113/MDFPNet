import os.path
import cv2
import logging
import time
import os

import numpy as np
from datetime import datetime
from collections import OrderedDict
from scipy.io import loadmat
#import hdf5storage
from scipy import ndimage
from scipy.signal import convolve2d

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util

from models.network_ufpnet_basic import UFPNet as net

from utils import utils_mat
import torch.profiler as profiler


'''
Spyder (Python 3.6)
PyTorch 1.4.0
Windows 10 or Linux
'''

"""
# --------------------------------------------
testing code of UFPNet
# --------------------------------------------

# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_name = '20000_G_stage7_20220610'
    testsetH_name = 'datasets_testSR/stand6/A_H'      # test set,  'set5' | 'srbsd68'
    testsetL_name = 'datasets_testSR/stand6/Input'    # Input_G0.002 , Input_S0.4
    # testsetH_name = 'datasets_testSR/stand6_Noise/A_H'      # test set,  'set5' | 'srbsd68'
    # testsetL_name = 'datasets_testSR/stand6_Noise/Input_G0.002'   #Input_G0.002 , Input_S0.4

    show_img = False           # default: False
    save_L = True              # save LR image
    save_E = True              # save estimated image

    n_channels = 1 #if 'gray' in  model_name else 3  # 3 for color image, 1 for grayscale image
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    result_name = testsetL_name + '_' + model_name
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path = H_path, E_path, logger
    # ----------------------------------------
    L_path = os.path.join(testsets, testsetL_name)  # L_path and H_path, fixed, for Low-quality images
    H_path = os.path.join(testsets, testsetH_name)
    E_path = os.path.join(results, result_name)    # E_path, fixed, for Estimated images
    util.mkdir(E_path)

    logger_name = model_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    model = net(n_iter=7, in_nc=1, out_nc=1).to(device)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for key, v in model.named_parameters():
        v.requires_grad = False
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    model = model.to(device)

    logger.info('Model path: {:s}'.format(model_path))
    logger.info('Params number: {}'.format(number_parameters))
    logger.info('Model_name:{}'.format(model_name))
    logger.info(L_path)

    L_paths = utils_mat.get_mat_paths(L_path)
    H_paths = util.get_image_paths(H_path)
    # --------------------------------
    # read images
    # --------------------------------
    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []
    test_results_ave['ssim'] = []

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    idx = 0

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_L1 = utils_mat.loadmat(img)
        img_L = img_L1['Y'].astype(np.float32)

        Ha_path = H_paths[idx]
        img_Ha = util.imread_uint(Ha_path, n_channels)

        img_Ha, img_L = util.uint2tensor3(img_Ha), util.single2tensor3(img_L)

        img_name, ext = os.path.splitext(os.path.basename(img))

        util.imshow(util.single2uint(img_L)) if show_img else None

        x = img_L

        [x] = [el.to(device) for el in [x]]

        x = torch.unsqueeze(x,0)
        # --------------------------------
        # (2) inference
        # --------------------------------
        x = model(x)

        # with profiler.profile(model(), x=(x,)) as prof:
        #     x = model(x)

        # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
        # --------------------------------
        # (3) img_E
        # --------------------------------
        img_E = util.tensor2uint(x[-1])
        # img_E = util.tensor2uint(x)

        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))

        # save the low resolution image (center image)
        if save_L:
            # img_Lc = util.tensor2uint(img_L[:,:,25])
            img_Lc = util.tensor2uint(img_L[24, :, :]/torch.max(img_L[24, :, :]))
            util.imsave(img_Lc, os.path.join(E_path, img_name+'_LR.png'))

        if need_H:
            # --------------------------------
            # (5) img_H
            # --------------------------------

            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = np.squeeze(img_H, 2)

            psnr = util.calculate_psnr(img_E, img_H)  # change with your own border
            ssim = util.calculate_ssim(img_E, img_H)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:->4d}--> {:>10s} PSNR: {:.2f}dB; SSIM: {:.4f}.'.format(idx, img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('------> Average PSNR(RGB) of ({}) {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))
    test_results_ave['psnr'].append(ave_psnr)
    test_results_ave['ssim'].append(ave_ssim)
    logger.info(test_results_ave['psnr'])
    logger.info(test_results_ave['ssim'])


if __name__ == '__main__':

    main()


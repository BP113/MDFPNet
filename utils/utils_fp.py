# -*- coding: utf-8 -*-
from utils import utils_image as util
import numpy as np


"""
# --------------------------------------------
# Fourier Ptychography
# --------------------------------------------
#
# Baopeng Li (libaopeng@opt.ac.cn)
# 03/03/2022
# --------------------------------------------
"""

"""
# --------------------------------------------
# the linear transform on the signal xx (namely "A").
# --------------------------------------------
"""

# Output: xx_c: n1_LR * n2_LR * L, sampling output (without abs).
# Input:  xx: n1 * n2, original signal (HR spectrum);
#         Masks: L * 2 (each point indicates the index of the left-upper point of the LR image in the HR spectrum);
#         n1_LR and n2_LR are the pixel numbers of xx_c (LR) in two dimensions;
#         pupil: the pupil function.

def A_LinearOperator(xx, Masks, pupil, n1_LR, n2_LR):
  L = Masks.shape[0]
  xx_c = np.zeros((n1_LR, n2_LR, L), dtype = "complex_")
  xx = np.squeeze(xx, 2)
  for k in range(0, L): #L-1
    index_x = Masks[k, 0]
    index_y = Masks[k, 1]
#    print(index_x,index_x+n1_LR-1,index_y)
    xx_c[:,:,k] = xx[index_x-1 : index_x+n1_LR-1, index_y-1 : index_y+n2_LR-1] * pupil
#  xx_c = np.fft.ifftshift(xx_c)

  xx_c = np.fft.ifftshift(xx_c, axes=(0, 1))
  yy_c = np.fft.ifft2(xx_c, axes=(0, 1))
  return yy_c


"""
# --------------------------------------------
# The inverse linear transform on the signal xx_c.
# --------------------------------------------
"""

# Output: xx_mean_large: n1 * n2, original signal (HR spectrum).
# Input:  xx_c: n1_LR * n2_LR * L, sampling output (without abs)
#         Masks: L * 2 (each point indicates the index of the left-upper point of the LR image in the HR spectrum)
#         n1 and n2 are the pixel numbers of xx_mean_large (HR) in two dimensions
#         pupil: the pupil function.

def A_Inverse_LinearOperator(yy_c, Masks, pupil, n1, n2):
  [n1_LR, n2_LR, L] = yy_c.shape
  xx_c = np.fft.fft2(yy_c, axes=(0,1))
  xx_c = np.fft.fftshift(xx_c, axes=(0,1))

  xx = np.zeros(shape = (max(Masks[:,0])-min(Masks[:,0])+n1_LR, max(Masks[:,1])-min(Masks[:,1])+n2_LR, L), dtype = "complex_") # save space
#  print('xx shape', xx.shape)
  for k in range(0, L):
    index_x = Masks[k, 0]
    index_y = Masks[k, 1]
    xx[index_x-min(Masks[:,0]):index_x-min(Masks[:,0])+n1_LR, index_y-min(Masks[:,1]):index_y-min(Masks[:,1])+n2_LR, k] = xx_c[:,:,k] * np.conj(pupil)

  xx_mean = xx.mean(axis = 2)
#  print('xx_mean shape', xx_mean.shape)
  xx_mean_large = np.zeros((n1, n2), dtype = "complex_")
#  print('xx mean large size:', xx_mean_large.shape)
#  print('xx mean size:', xx_mean.shape[0])
#  print(Masks[:,0].min(),Masks[:,0].min()+xx_mean.shape[0]+1)
  xx_mean_large[Masks[:,0].min()-1 : Masks[:,0].min()+xx_mean.shape[0]-1, Masks[:,1].min()-1 : Masks[:,1].min()+xx_mean.shape[1]-1] = xx_mean
  return xx_mean_large



if __name__ == '__main__':
    img = util.imread_uint('test.bmp', 3)

    img = util.uint2single(img)
    # k = anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6)
    util.imshow(img)

    
    
    
    
    
    

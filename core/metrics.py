import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid


def tensor2img(tensor, out_type=np.uint16, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu() # clamp
    # tensor = (tensor - min_max[0]) / \
    #     (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    return img_np


def save_img(img, img_path, mode='RGB'):
    #cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img=img = img[:, :, np.newaxis]
    cv2.imwrite(img_path, img[:,:])


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    return mse

def ssim(img1, img2):
    pass


def calculate_ssim(img1, img2):
   pass
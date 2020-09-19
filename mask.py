import torch.nn as nn
import torch
import cv2
import os
import numpy as np
from DoubleDIP.utils.image_io import *



dataRoot = '/home/r/shangwei/defence/zipaigan/'#'/media/r/dataset/datasets/dataset/Test_Set/Test_Images/'  # fenced images
maskRoot = '/home/r/shangwei/defence/zipaigan/scribble/' #'/media/r/dataset/datasets/dataset/Test_Set/Test_Labels/'  # masks
Savepath = '/home/r/shangwei/defence/zipaigan/mask/'  # results of bid




def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False

image = prepare_image(dataRoot + '2.jpg')
scribble = prepare_image(maskRoot + '2.jpg')
hint = np.sign(np.sum(scribble - image, axis=0, keepdims=True)) / 2 + 0.5

hint = hint!=0.5
hint = hint.astype(float)

save_image("2", hint, Savepath)












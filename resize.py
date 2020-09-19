import torch.nn as nn
import torch
import cv2
import os
import numpy as np
from BidAttenMap.test_random import bid
from DoubleDIP.segmentation import segmentation
from DoubleDIP.segmentation_nomasknet import segmentation_nomask
from DoubleDIP.utils.image_io import *
from PReNet.test_PReNet import Predict


bid_size = (896, 512)#(1664, 1280)  # h, w  #the size of pictures in bid, and it must be an integral multiple of 128.


dataRoot = '/media/r/dataset/datasets/dataset/Test_Set/Test_Images/' #'/home/r/shangwei/defence/zipaigan/'  # fenced images

# the path of preprocessed masks
Maskroot = '/media/r/dataset/datasets/dataset/Test_Set/test_bidsize/'
os.makedirs(Maskroot, exist_ok=True)

def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False



for img_name in os.listdir(dataRoot):
    if is_image(img_name):

        img_path = os.path.join(dataRoot, img_name)

        y1 = cv2.imread(img_path)
        im = cv2.resize(y1, bid_size, interpolation=cv2.cv2.INTER_AREA)           
        cv2.imwrite(os.path.join(Maskroot, img_name), im)
            

         















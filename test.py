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

prepare = "True"  # if you need to preprocess mask
predict = "True"  # if you use the mask predicted by PReNet, otherwise 'False' indicates the label mask.
dilate_scale = 10
bid_size = (1664, 1280)#(512, 896)  # h, w  #the size of pictures in bid, and it must be an integral multiple of 128.
dip_size = (1632, 920)  # w, h  #the size of pictures in dip, dip can process it in an appropriate value.
method = 'dip'  # dip or bid

dataRoot = '/media/r/dataset/datasets/dataset/Test_Set/Test_Images/' #'/home/r/shangwei/defence/zipaigan/'  # fenced images
maskRoot = '/home/r/shangwei/PReNet-defence/results/PReNet4_defence_L1+0.1nonblurry_gray+edge+asymmetric0.2/' #'/media/r/dataset/datasets/dataset/Test_Set/Test_Labels/'  # masks
Savepath = '/home/r/shangwei/defence/results_bid_zipaigan/predict_' + predict + '/'  # results of bid

# the path of preprocessed masks
Maskroot = '/media/r/dataset/datasets/dataset/Test_Set/biddirection/predict_' + predict + '/Test_Labels_%d/' % dilate_scale #/home/r/shangwei/defence/zipaigan/rmask/
os.makedirs(Maskroot, exist_ok=True)

logRoot = '/home/r/shangwei/defence/PReNet/logs/'  # the path of pretrained PReNet model
predict_path = '/home/r/shangwei/defence/predicted_mask'  # the path of predicted masks
def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False


if method == 'bid':
    if predict == "False":
        if prepare == "False":

            print("Masks in bid must be processed !")
        else:
            for img_name in os.listdir(maskRoot):
                if is_image(img_name):

                    img_path = os.path.join(maskRoot, img_name)

                    y1 = cv2.imread(img_path, 0)
                    y1 = y1 / 255.0
                    y1[y1 <= 0.4] = 0
                    y1[y1 > 0.4] = 1
                    y1 = y1 * 255.0
                    #kernel = np.ones((dilate_scale, dilate_scale), np.uint8)
                    #y1 = cv2.dilate(y1, kernel, iterations=1)
                    y = 255 - y1
                    cv2.imwrite(Maskroot + img_name, y)
            bid(dataRoot, Maskroot, Savepath, bid_size)
    else:
        Predict(dataRoot=dataRoot, logRoot=logRoot, savepath=predict_path)
        maskRoot = predict_path
        if prepare == "False":

            print("Masks in bid must be processed !")
        else:
            for img_name in os.listdir(maskRoot):
                if is_image(img_name):
                    img_path = os.path.join(maskRoot, img_name)

                    y1 = cv2.imread(img_path, 0)
                    y1 = y1 / 255.0
                    y1[y1 <= 0.4] = 0
                    y1[y1 > 0.4] = 1
                    y1 = y1 * 255.0
                    kernel = np.ones((dilate_scale, dilate_scale), np.uint8)
                    y1 = cv2.dilate(y1, kernel, iterations=1)
                    y = 255 - y1
                    cv2.imwrite(Maskroot + img_name, y)
            bid(dataRoot, Maskroot, Savepath, bid_size)

elif method == 'dip':
    if predict == "False":
        ids = [0]
        for imgid in range(1, 101):

            if imgid in ids:
                continue
            else:
                im = cv2.imread(dataRoot + '2017_Test_00%03d.jpg' % imgid, cv2.IMREAD_COLOR)
                fore = cv2.imread(maskRoot + '2017_Test_00%03d.png' % imgid, cv2.IMREAD_COLOR)
                im1 = cv2.resize(im, dip_size, interpolation=cv2.cv2.INTER_AREA)
                fore1 = cv2.resize(fore, dip_size, interpolation=cv2.INTER_AREA)

                load_path = '/home/r/shangwei/defence/images/fence/%d/' % imgid
                os.makedirs(load_path, exist_ok=True)

                imgname = '/home/r/shangwei/defence/images/fence/%d/%03d.jpg' % (imgid, imgid)

                cv2.imwrite(imgname, im1)
                cv2.imwrite('/home/r/shangwei/defence/images/fence/%d/%03d.png' % (imgid, imgid), fore1)


                im = prepare_image(imgname)
                if prepare == "True":
                    imgname = '/home/r/shangwei/defence/images/fence/%d/%03d.png' % (imgid, imgid)
                    img = cv2.imread(imgname, 0)
                    kernel = np.ones((dilate_scale, dilate_scale), np.uint8)
                    fore1 = cv2.dilate(img, kernel, iterations=1)
                    imgname = '/home/r/shangwei/defence/images/fence/%d/%03d-%d.png' % (imgid, imgid, dilate_scale)
                    cv2.imwrite(imgname, fore1)
                    save_path = '/home/r/shangwei/defence/results_dip/nomasknet/%d-nomasknet/' % imgid
                else:
                    imgname = '/home/r/shangwei/defence/images/fence/%d/%03d.png' % (imgid, imgid)
                    save_path = '/home/r/shangwei/defence/results_dip/nomasknet_nodilation/%d-nomasknet/' % imgid

                fore = prepare_image(imgname)
                fore[fore <= 0.4] = 0
                fore[fore > 0.4] = 1

                os.makedirs(save_path, exist_ok=True)
                save_image("mask_target", fore, save_path)
                back = 1 - fore
                segmentation_nomask('%d_image' % imgid, im, fore, back, save_path)
    else:
        ids = [0]
        for imgid in range(1, 101):
            #Predict(dataRoot, logRoot, predict_path)
            maskRoot = maskRoot#predict_path
            if imgid in ids:
                continue
            else:
                im = cv2.imread(dataRoot + '2017_Test_00%03d.jpg' % imgid, cv2.IMREAD_COLOR)
                fore = cv2.imread(maskRoot + '2017_Test_00%03d.png' % imgid, cv2.IMREAD_COLOR)
                im1 = cv2.resize(im, dip_size, interpolation=cv2.cv2.INTER_AREA)
                fore1 = cv2.resize(fore, dip_size, interpolation=cv2.INTER_AREA)

                load_path = '/home/r/shangwei/defence/images/fence/%d/' % imgid
                os.makedirs(load_path, exist_ok=True)

                imgname = '/home/r/shangwei/defence/images/fence/%d/%03d.jpg' % (imgid, imgid)

                cv2.imwrite(imgname, im1)
                cv2.imwrite('/home/r/shangwei/defence/images/fence/%d/%03d.png' % (imgid, imgid), fore1)

                im = prepare_image(imgname)
                if prepare == "True":
                    imgname = '/home/r/shangwei/defence/images/fence/%d/%03d.png' % (imgid, imgid)
                    img = cv2.imread(imgname, 0)
                    kernel = np.ones((dilate_scale, dilate_scale), np.uint8)
                    fore1 = cv2.dilate(img, kernel, iterations=1)
                    imgname = '/home/r/shangwei/defence/images/fence/%d/%03d-%d.png' % (imgid, imgid, dilate_scale)
                    cv2.imwrite(imgname, fore1)
                    save_path = '/home/r/shangwei/defence/results_dip/masknet/%d-masknet/' % imgid
                else:
                    imgname = '/home/r/shangwei/defence/images/fence/%d/%03d.png' % (imgid, imgid)
                    save_path = '/home/r/shangwei/defence/results_dip/masknet_nodilation/%d-masknet/' % imgid

                fore = prepare_image(imgname)
                fore[fore <= 0.4] = 0
                fore[fore > 0.4] = 1


                os.makedirs(save_path, exist_ok=True)
                save_image("mask_target", fore, save_path)
                back = 1 - fore
                segmentation('%d_image' % imgid, im, fore, back, save_path)















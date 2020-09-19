import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *
import random

def Im2Patch(img, win, stride=1):
    k = 0
    #endc = img.shape[0]
    endw = img.shape[0]
    endh = img.shape[1]
    patch = img[0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[0] * patch.shape[1]
    Y = np.zeros([win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[k, :] = np.array(patch[:]).reshape(1, TotalPatNum)
            k = k + 1
    return Y.reshape([win, win, TotalPatNum])



# def prepare_data_defence(data_path, patch_size, stride):
#     # train
#     print('process training data')
#     input_path = os.path.join(data_path, 'Training_Images')
#     target_path = os.path.join(data_path, 'Training_Labels')
#
#     save_target_path = os.path.join(data_path, 'Training_Labels', 'train_target.h5')
#     save_input_path = os.path.join(data_path, 'Training_Images', 'train_input.h5')
#
#     target_h5f = h5py.File(save_target_path, 'w')
#     input_h5f = h5py.File(save_input_path, 'w')
#
#     train_num = 0
#     for i in range(545):
#         target_file = "2017_Train_%05d.png" % (i + 1)
#         target1 = cv2.imread(os.path.join(target_path, target_file), cv2.IMREAD_GRAYSCALE)
#         target = cv2.resize(target1, (908, 512), interpolation=cv2.INTER_AREA)
#         #print(target.shape)  #512,908,3
#         # b, g, r = cv2.split(target)
#         # target = cv2.merge([r, g, b])
#
#         for j in range(3):
#             input_file = "2017_Train_%05d.jpg" % (i + 1)
#             input_img1 = cv2.imread(os.path.join(input_path,input_file))
#             input_img2 = cv2.cvtColor(input_img1, cv2.COLOR_RGB2GRAY)
#             #input_img1 = input_img2.reshape((input_img2[0], input_img2[1], 1))
#             #input_img1 = np.array([input_img2],[input_img2],[input_img2])
#
#             #input_img1 = input_img1.transpose(1, 2, 0)
#             #print(input_img2.shape)
#             input_img = cv2.resize(input_img2, (908, 512), interpolation=cv2.INTER_AREA)
#             #print(input_img.shape)
#             #print(input_img.shape)
#             #print(os.path.join(input_path,input_file))
#             # b, g, r = cv2.split(input_img)
#             # input_img = cv2.merge([r, g, b])
#
#             target_img = target
#
#             if j == 1:
#                 target_img = cv2.resize(target1, (1632, 920), interpolation=cv2.INTER_AREA)
#                 input_img = cv2.resize(input_img2, (1632, 920), interpolation=cv2.INTER_AREA)
#
#             if j == 2:
#                 target_img = cv2.resize(target1, (816, 460), interpolation=cv2.INTER_AREA)
#                 input_img = cv2.resize(input_img2, (816, 460), interpolation=cv2.INTER_AREA)
#             #print(input_img.shape)#512,908
#             target_img = np.float32(normalize(target_img))
#             target_patches = Im2Patch(target_img, win=patch_size, stride=stride)
#
#             input_img = np.float32(normalize(input_img))
#             input_patches = Im2Patch(input_img, win=patch_size, stride=stride)
#
#             print("target file: %s # samples: %d" % (input_file, target_patches.shape[2]))
#             for n in range(target_patches.shape[2]):
#                 target_data = target_patches[:, :, n].copy()
#                 target_h5f.create_dataset(str(train_num), data=target_data)
#
#                 input_data = input_patches[:, :, n].copy()
#                 input_h5f.create_dataset(str(train_num), data=input_data)
#
#                 train_num += 1
#
#     target_h5f.close()
#     input_h5f.close()
#
#     print('training set, # samples %d\n' % train_num)


# def prepare_data_defence(data_path, patch_size, stride):
#     # train
#     print('process training data')
#     input_path = os.path.join(data_path, 'Training_Images')
#     target_path = os.path.join(data_path, 'Training_Labels')
#
#     save_target_path = os.path.join(data_path, 'Training_Labels', 'train_target.h5')
#     save_input_path = os.path.join(data_path, 'Training_Images', 'train_input.h5')
#     save_edge_path = os.path.join(data_path, 'Training_Images', 'edge_input.h5')
#
#     target_h5f = h5py.File(save_target_path, 'w')
#     input_h5f = h5py.File(save_input_path, 'w')
#     edge_h5f = h5py.File(save_edge_path, 'w')
#
#     train_num = 0
#     for i in range(545):
#
#
#         for j in range(3):
#             target_file = "2017_Train_%05d.png" % (i + 1)
#             target = cv2.imread(os.path.join(target_path, target_file))
#             target1 = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
#             target = cv2.resize(target1, (656, 656), interpolation=cv2.INTER_AREA)
#             # print(target.shape)  #512,908,3
#             # b, g, r = cv2.split(target)
#             # target = cv2.merge([r, g, b])
#             h1, w1 = target.shape[:2]
#             center1 = (w1 // 2, h1 // 2)
#             # s = random.randint(1, 5)
#
#             input_file = "2017_Train_%05d.jpg" % (i + 1)
#             input_img1 = cv2.imread(os.path.join(input_path,input_file))
#             input_img2 = cv2.cvtColor(input_img1, cv2.COLOR_RGB2GRAY)
#             #input_img1 = input_img2.reshape((input_img2[0], input_img2[1], 1))
#             #input_img1 = np.array([input_img2],[input_img2],[input_img2])
#
#             #input_img1 = input_img1.transpose(1, 2, 0)
#             #print(input_img2.shape)
#             input_img = cv2.resize(input_img2, (656, 656), interpolation=cv2.INTER_AREA)
#             edge_img = cv2.Canny(input_img, 50, 150)
#             h2, w2 = input_img.shape[:2]
#             center2 = (w2 // 2, h2 // 2)
#             #print(input_img.shape)
#
#             #print(os.path.join(input_path,input_file))
#             # b, g, r = cv2.split(input_img)
#             # input_img = cv2.merge([r, g, b])
#
#             # if i % 2 == 0:
#             if j > 0:
#                 M1 = cv2.getRotationMatrix2D(center1, -90 // (j+1), 1)
#                 target = cv2.warpAffine(target, M1, (h1, w1))
#                 M2 = cv2.getRotationMatrix2D(center2, -90 // (j + 1), 1)
#                 input_img = cv2.warpAffine(input_img, M2, (h2, w2))
#                 edge_img = cv2.Canny(input_img, 50, 150)
#
#
#                 target = target[int(h1 // 2 * (1 - 1 / 2 ** 0.5)):int(h1 // 2 * (1 + 1 / 2 ** 0.5)),
#                             int(w1 // 2 * (1 - 1 / 2 ** 0.5)):int(w1 // 2 * (1 + 1 / 2 ** 0.5))]
#                 input_img = input_img[int(h2 // 2 * (1 - 1 / 2 ** 0.5)):int(h2 // 2 * (1 + 1 / 2 ** 0.5)),
#                       int(w2 // 2 * (1 - 1 / 2 ** 0.5)):int(w2 // 2 * (1 + 1 / 2 ** 0.5))]
#                 edge_img = edge_img[int(h2 // 2 * (1 - 1 / 2 ** 0.5)):int(h2 // 2 * (1 + 1 / 2 ** 0.5)),
#                       int(w2 // 2 * (1 - 1 / 2 ** 0.5)):int(w2 // 2 * (1 + 1 / 2 ** 0.5))]
#
#
#
#             target_img = target
#
#
#             #print(input_img.shape)#512,908
#             target_img = np.float32(normalize(target_img))
#             target_patches = Im2Patch(target_img, win=patch_size, stride=stride)
#
#             input_img = np.float32(normalize(input_img))
#             input_patches = Im2Patch(input_img, win=patch_size, stride=stride)
#
#             edge_img = np.float32(normalize(edge_img))
#             edge_patches = Im2Patch(edge_img, win=patch_size, stride=stride)
#
#             print("target file: %s # samples: %d" % (input_file, target_patches.shape[2]))
#             for n in range(target_patches.shape[2]):
#                 target_data = target_patches[:, :, n].copy()
#                 target_h5f.create_dataset(str(train_num), data=target_data)
#
#                 input_data = input_patches[:, :, n].copy()
#                 input_h5f.create_dataset(str(train_num), data=input_data)
#
#                 edge_data = edge_patches[:, :, n].copy()
#                 edge_h5f.create_dataset(str(train_num), data=edge_data)
#
#                 train_num += 1
#
#     target_h5f.close()
#     input_h5f.close()
#     edge_h5f.close()
#
#     print('training set, # samples %d\n' % train_num)

def prepare_data_defence(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'Training_Images')
    target_path = os.path.join(data_path, 'Training_Labels')

    save_target_path = os.path.join(data_path, 'Training_Labels', 'train_target.h5')
    save_input_path = os.path.join(data_path, 'Training_Images', 'train_input.h5')
    save_edge_path = os.path.join(data_path, 'Training_Images', 'edge_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')
    edge_h5f = h5py.File(save_edge_path, 'w')

    train_num = 0
    for i in range(545):

        target_file = "2017_Train_%05d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path, target_file))
        target1 = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        target = cv2.resize(target1, (908, 512), interpolation=cv2.INTER_AREA)
        kernel2 = np.ones((5, 5), np.uint8)
        target = cv2.dilate(target, kernel2, iterations=1)

        input_file = "2017_Train_%05d.jpg" % (i + 1)
        input_img1 = cv2.imread(os.path.join(input_path,input_file))
        input_img2 = cv2.cvtColor(input_img1, cv2.COLOR_RGB2GRAY)

        input_img = cv2.resize(input_img2, (908, 512), interpolation=cv2.INTER_AREA)
        edge_img = cv2.Canny(input_img, 50, 150)
        kernel1 = np.ones((8, 8), np.uint8)
        edge_img = cv2.dilate(edge_img, kernel1, iterations=1)

        target_img = target

        target_img = np.float32(normalize(target_img))
        target_patches = Im2Patch(target_img, win=patch_size, stride=stride)

        input_img = np.float32(normalize(input_img))
        input_patches = Im2Patch(input_img, win=patch_size, stride=stride)

        edge_img = np.float32(normalize(edge_img))
        edge_patches = Im2Patch(edge_img, win=patch_size, stride=stride)

        print("target file: %s # samples: %d" % (input_file, target_patches.shape[2]))
        for n in range(target_patches.shape[2]):
            target_data = target_patches[:, :, n].copy()
            target_h5f.create_dataset(str(train_num), data=target_data)

            input_data = input_patches[:, :, n].copy()
            input_h5f.create_dataset(str(train_num), data=input_data)

            edge_data = edge_patches[:, :, n].copy()
            edge_h5f.create_dataset(str(train_num), data=edge_data)

            train_num += 1

    target_h5f.close()
    input_h5f.close()
    edge_h5f.close()

    print('training set, # samples %d\n' % train_num)


class Dataset(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset, self).__init__()

        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'Training_Labels', 'train_target.h5')
        input_path = os.path.join(self.data_path, 'Training_Images', 'train_input.h5')
        edge_path = os.path.join(self.data_path, 'Training_Images', 'edge_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')
        edge_h5f = h5py.File(edge_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()
        edge_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, 'Training_Labels', 'train_target.h5')
        input_path = os.path.join(self.data_path, 'Training_Images', 'train_input.h5')
        edge_path = os.path.join(self.data_path, 'Training_Images', 'edge_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')
        edge_h5f = h5py.File(edge_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])
        edge = np.array(edge_h5f[key])

        target_h5f.close()
        input_h5f.close()
        edge_h5f.close()

        return torch.Tensor(input), torch.Tensor(target), torch.Tensor(edge)



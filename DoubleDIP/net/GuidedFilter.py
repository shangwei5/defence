# This tensorflow implementation of guided filter is from: H. Wu, et al., "Fast End-to-End Trainable Guided Filter", CPVR, 2018.

# Web: https://github.com/wuhuikai/DeepGuidedFilter

#import torch
import numpy as np
import cv2
from utils.image_io import *
# def diff_x(input, r):
#     assert input.dim() == 4
#
#     left   = input[:, :,         r:2 * r + 1]
#     middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
#     right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]
#
#     output = torch.cat([left, middle, right], 2)
#
#     return output
#
#
# def diff_y(input, r):
#     assert input.dim() == 4
#
#     left   = input[:, :, :,         r:2 * r + 1]
#     middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
#     right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]
#
#     output = torch.cat([left, middle, right], 3)
#
#     return output
#
#
# def box_filter(x, r):
#     assert x.dim() == 4
#
#     return diff_y(torch.cumsum(diff_x(torch.cumsum(x, dim=2), r), dim=3), r)  #cumsum是求累积和函数
#
#
# def guided_filter(x, y, r, eps=1e-8, nhwc=False):
#     assert x.dim() == 4 and y.dim() == 4
#
#
#     # data format
#     if nhwc:
#         x = torch.transpose(x, [0, 3, 1, 2])
#         y = torch.transpose(y, [0, 3, 1, 2])
#
#     # shape check
#     x_shape = x.shape
#     y_shape = y.shape
#
#     #assets = [torch.assert_equal(   x_shape[0],  y_shape[0]),
#      #         torch.assert_equal(  x_shape[2:], y_shape[2:]),
#      #         torch.assert_greater(x_shape[2:],   2 * r + 1),
#       #        torch.Assert(torch.logical_or(torch.equal(x_shape[1], 1),
#        #                                     torch.equal(x_shape[1], y_shape[1])), [x_shape, y_shape])]
#
#    # with torch.control_dependencies(assets):
#     #    x = torch.identity(x)
#
#     # N
#     N = box_filter(torch.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r)  #dtype是数据类型
#
#     # mean_x
#     mean_x = box_filter(x, r) / N
#     # mean_y
#     mean_y = box_filter(y, r) / N
#     # cov_xy
#     cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
#     # var_x
#     var_x  = box_filter(x * x, r) / N - mean_x * mean_x
#
#     # A
#     A = cov_xy / (var_x + eps)
#     # b
#     b = mean_y - A * mean_x
#
#     mean_A = box_filter(A, r) / N
#     mean_b = box_filter(b, r) / N
#
#     output = mean_A * x + mean_b
#
#     if nhwc:
#         output = torch.transpose(output, [0, 2, 3, 1])
#
#     return output

def guided_filter(data, guide, num_patches = 1, width = None, height = None, channel = None, r = 30, eps = 1e-4):
     #15
    #1.0
    batch_q = np.zeros((num_patches, height, width, channel))
    data = data.permute(0,2,3,1)
    guide = guide.permute(0, 2, 3, 1)
    ch1 = data.shape[3]
    ch2 = guide.shape[3]

    #data = torch_to_np(data)

    #print(num_patches)
    #print(height)
    #print(width)
    #print(channel)
    for i in range(num_patches):
        for j in range(channel):
            if ch1>1:
                I = data[i, :, :,j]
                I = I.expand(1, I.shape[0], I.shape[1])
                I = torch_to_np(I)
            #print(I.shape)
            else:
                I = data[i, :, :, 0]
                I = I.expand(1, I.shape[0], I.shape[1])
                I = torch_to_np(I)
            if ch2 >1:
                p = guide[i, :, :,j]
                p = p.expand(1, p.shape[0], p.shape[1])
                p = torch_to_np(p)
            else:
                p = guide[i, :, :, 0]
                p = p.expand(1, p.shape[0], p.shape[1])
                p = torch_to_np(p)
            #print(p.shape)
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            q = mean_a * I + mean_b
            batch_q[i, :, :,j] = q
    return batch_q
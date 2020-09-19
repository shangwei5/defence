import torch
from torch import nn
import numpy as np
# from .layers import bn, VarianceLayer, CovarianceLayer, GrayscaleLayer
# from .downsampler import *
# from .closed_form_matting import *
from torch.nn import functional
# from .gmm import *
# from .gmm2 import *
import cv2
# from .closed_form_matting import *
# from net.skip_model import VGG16FeatureExtractor



class ExtendedL1Loss(nn.Module):
    """
    also pays attention to the mask, to be relative to its size
    """
    def __init__(self):
        super(ExtendedL1Loss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, b, mask):
        normalizer = self.l1(mask, torch.zeros(mask.shape).cuda())
        # if normalizer < 0.1:
        #     normalizer = 0.1
        c = self.l1(mask * a, mask * b) / normalizer
        return c


class NonBlurryLoss(nn.Module):
    def __init__(self):
        """
        Loss on the distance to 0.5
        """
        super(NonBlurryLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x):
        return 1 - self.mse(x, torch.ones_like(x) * 0.5)


class TVLoss(nn.Module):
    def __init__(self,loss_weight=1):
        super(TVLoss,self).__init__()
        self.loss_weight = loss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.loss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class asymmetricLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super(asymmetricLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x, y): # x is output, y is target
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        m = torch.zeros(x.shape)
        for i in range(0, batch_size):
            for j in range(0, h_x):
                for k in range(0, w_x):
                    if x[i, 0, j, k] < y[i, 0, j, k]:
                        m[i, 0, j, k] = 1
                    else:
                        m[i, 0, j, k] = 0
        map = torch.abs(self.alpha - m)

        return self.l1(map * x, map * y)

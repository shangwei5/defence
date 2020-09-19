import torch
from torch import nn
import numpy as np
from .layers import bn, VarianceLayer, CovarianceLayer, GrayscaleLayer
from .downsampler import *
from .closed_form_matting import *
from torch.nn import functional
from .gmm import *
from .gmm2 import *
import cv2
#from .closed_form_matting import *
from DoubleDIP.net.skip_model import VGG16FeatureExtractor

class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))


class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(1):
            for j in range(1):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


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


class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.gray_scale = GrayscaleLayer()
        self.mse = nn.MSELoss().cuda()

    def forward(self, x, y):
        x_g = self.gray_scale(x)
        y_g = self.gray_scale(y)
        return 1 / self.mse(x_g, y_g)


class GrayLoss(nn.Module):
    def __init__(self):
        super(GrayLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x):
        y = torch.ones_like(x) / 2.
        return 1 / self.l1(x, y)


class GradientLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)


class YIQGNGCLoss(nn.Module):
    def __init__(self, shape=5):
        super(YIQGNGCLoss, self).__init__()
        self.shape = shape
        self.var = VarianceLayer(self.shape, channels=1)
        self.covar = CovarianceLayer(self.shape, channels=1)

    def forward(self, x, y):
        if x.shape[1] == 3:
            x_g = rgb_to_yiq(x)[:, :1, :, :]  # take the Y part
            y_g = rgb_to_yiq(y)[:, :1, :, :]  # take the Y part
        else:
            assert x.shape[1] == 1
            x_g = x  # take the Y part
            y_g = y  # take the Y part
        c = torch.mean(self.covar(x_g, y_g) ** 2)
        vv = torch.mean(self.var(x_g) * self.var(y_g))
        return c / vv

# class EnergyLoss(nn.Module):
#     def __init__(self):
#         super(EnergyLoss, self).__init__()
#         #self.l1 = nn.L1Loss().cuda()
#
#     def forward(self, x, y, p, z):
#
#         e1 = x * compute_alpha(z) #.cuda()
#         #print(e1.shape)  # #(510*510*9*9=21068100)
#         z = torch_to_np(z)
#         # X = np.einsum('...ij,...jk->...ik', z.ravel() - p.ravel(), y)
#         # e2 = np.einsum('...ij,...kj->...ik', X, z.ravel() - p.ravel())
#         e2 = (z.ravel() - p.ravel()) * y * (z.ravel() - p.ravel())
#         #print(e2.shape)  #(262144,)
#         #print(y.shape)
#         e = e1.mean() + e2.mean()
#         #print(e.shape)
#         return e


class EnergyLoss(nn.Module):
    def __init__(self):
        super(EnergyLoss, self).__init__()
        #self.l1 = nn.L1Loss().cuda()

    def forward(self, x, c, y, p, z):

        e1 = x * compute_alpha(z, c) #.cuda()
        z = torch_to_np(z)
        e2 = (y + p) * (z - y)**2
        e = e1.mean() + e2.mean()
        return e

class EnergyLoss_rain(nn.Module):
    def __init__(self):
        super(EnergyLoss_rain, self).__init__()
        #self.l1 = nn.L1Loss().cuda()

    def forward(self, x, y, z):

        e1 = x * compute_alpha(z) #.cuda()
        z = torch_to_np(z)
        y = torch_to_np(y)
        e2 = (z - y)**2
        e = e1.mean() + 1e-4 * e2.mean()
        return e


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


class GMMLoss(nn.Module):
    def __init__(self, k_components = 3):
        super(GMMLoss, self).__init__()
        #self.k_components = k_components

    def forward(self, x,gmm):
        x = x.view([-1,1])
        #d = x.shape[1]
        #print(d)

        gmm.fit(x)
        y = gmm.predict(x)
        #print(y.shape)
        return torch.mean(y)


class GMMLoss1(nn.Module):
    def __init__(self, k_components = 3):
        super(GMMLoss1, self).__init__()
        self.k_components = k_components
        self.loss = 0

    def forward(self, x, y, step, model):  #x为real data;y为image_out
        #print(x.shape)
        x = torch_to_np(x)
        y = torch_to_np(y)
        #print(x.reshape(1, -1).shape)
        #print(y.reshape(1, -1).shape)
        if step==0:

            Error = x.reshape(1, -1)
        else:
            #unsupervised_outputs = y
            Error = x.reshape(1, -1) - y.reshape(1, -1)
        R_real, _ = expectation(Error, model)
        model = maximizationModel(Error, R_real)
        R_real, _ = expectation(Error, model)

        for k in range(self.k_components):
            #print(R_real[:, k].shape)
            #print(np.square(x - y).shape)
            self.loss += R_real[:, k] * np.square(x - y).reshape(1, -1) * .5 / model['Sigma'][0, k]
            #print(self.loss.shape)
        loss = np.mean(self.loss)

        return loss



def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)  #bmm  batch matrix multiply
    return gram

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.extractor = VGG16FeatureExtractor()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x, y):  #x为input，y为output。
        if y.shape[1] == 3:
            feat_output_comp = self.extractor(x)
            feat_output = self.extractor(y)
        elif y.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([x]*3, 1))
            feat_output = self.extractor(torch.cat([y]*3, 1))
        else:
            raise ValueError('only gray an')
        loss = 0
        for i in range(3):
            loss += 0.05 * self.l1(feat_output[i], feat_output_comp[i])

        for i in range(3):
            loss += 12 * self.l1(gram_matrix(feat_output[i]),gram_matrix(feat_output_comp[i]))

        return loss


class ColourLoss(nn.Module):
    def __init__(self):
        super(ColourLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x):
        #y = torch.ones_like(x) * 1.5 / 2.
        y = x>0.6
        return 1/y.sum()#self.l1(x, y)
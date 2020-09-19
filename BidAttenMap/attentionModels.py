import math
import torch 
from torch import nn
from torchvision import models
#from spectral_normalization import SpectralNorm

#weight initial strategy
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__

        if (classname.find('Conv') == 0 or classname.find('Linear') == 0 ) and hasattr(m, 'weight'):
            if (init_type == 'gaussian'):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif (init_type == 'xavier'):
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif (init_type == 'kaiming'):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif (init_type == 'orthogonal'):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif (init_type == 'default'):
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

#VGG16 feature
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


#define adaptive learning partial conv
class LearningPartialConv(nn.Module):
    def __init__(self, inputChannels, outputChannels, kernelSize, stride, 
        padding, dilation=1, groups=1, bias=False, maskInit=0.05):
        super(LearningPartialConv, self).__init__()

        self.conv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, dilation, groups, bias)
        if inputChannels == 4:
            self.maskConv = nn.Conv2d(3, outputChannels, kernelSize, stride, padding, dilation, groups, bias)
        else:
            self.maskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init())

        nn.init.constant_(self.maskConv.weight, maskInit)

        self.activ = nn.Tanh()

    def forward(self, inputImage, mask):
        convFeatures = self.conv(inputImage)
        maskFeatures = self.maskConv(mask)

        maskActiv = self.activ(maskFeatures)
        maskActiv = torch.abs(maskActiv)       
        convOut = convFeatures * maskActiv

        return convOut, maskActiv

#define our adaptive learning partial layer
class AdaptivePConvDefine(nn.Module):
    def __init__(self, inputChannels, outputChannels, bn=True, sample='down-4', activ='leaky', convBias=False, maskInit=0.25):
        super().__init__()

        if sample == 'down-4':
            self.conv = LearningPartialConv(inputChannels, outputChannels, 4, 2, 1, bias=convBias, maskInit=maskInit)
        elif sample == 'down-5':
            self.conv = LearningPartialConv(inputChannels, outputChannels, 5, 2, 2, bias=convBias, maskInit=maskInit)
        elif sample == 'down-7':
            self.conv = LearningPartialConv(inputChannels, outputChannels, 7, 2, 3, bias=convBias, maskInit=maskInit)
        elif sample == 'down-3':
            self.conv = LearningPartialConv(inputChannels, outputChannels, 3, 2, 1, bias=convBias, maskInit=maskInit)
        else:
            self.conv = LearningPartialConv(inputChannels, outputChannels, 3, 1, 1, bias=convBias, maskInit=maskInit)
        
        if bn:
            self.bn = nn.BatchNorm2d(outputChannels)
        
        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass
    
    def forward(self, inputFeatures, inputMasks):

        features, maskFeatures = self.conv(inputFeatures, inputMasks)
        forConcat = features
        if hasattr(self, 'bn'):
            features = self.bn(features)
        if hasattr(self, 'activ'):
            features = self.activ(features)
         
        return forConcat, features, maskFeatures


#define opposite mask conv( conv for 1 - mask)
class OppoMaskConv(nn.Module):
    def __init__(self, inputChannels, outputChannels, kernelSize, stride, 
        padding, dilation=1, groups=1, convBias=False, maskInit=0.025):
        super().__init__()

        self.OppoMaskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, bias=convBias)

        nn.init.constant_(self.OppoMaskConv.weight, maskInit)

        self.activ = nn.Sigmoid()
    
    def forward(self, inputMasks):
        maskFeatures = self.OppoMaskConv(inputMasks)

        maskFeatures = self.activ(maskFeatures)

        return maskFeatures

#define up conv part
class UpAdaptiveConv(nn.Module):
    def __init__(self, inputChannels, outputChannels, bn=True, activ='leaky', 
        kernelSize=4, stride=2, padding=1, dilation=1, groups=1,convBias=False):
        super().__init__()

        self.conv = nn.ConvTranspose2d(inputChannels, outputChannels, kernel_size=kernelSize, 
            stride=stride, padding=padding, dilation=dilation, groups=groups,bias=convBias)
        
        self.conv.apply(weights_init())

        if bn:
            self.bn = nn.BatchNorm2d(outputChannels)
        
        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass
    
    def forward(self, forConcat, inputFeatures, maskFeatures):
        featuresFirst = self.conv(inputFeatures)

        if hasattr(self, 'bn'):
            featuresFirst = self.bn(featuresFirst)
        concat = torch.cat((forConcat, featuresFirst),  1)
        outFeatures = concat * maskFeatures
        if hasattr(self, 'activ'):
            outFeatures = self.activ(outFeatures)
        
        return outFeatures

#define Combined Model
class CombinedNet(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super().__init__()

        self.ec1 = AdaptivePConvDefine(inputChannels, 64, bn=False, sample='down-4')
        self.ec2 = AdaptivePConvDefine(64, 128, sample='down-4')
        self.ec3 = AdaptivePConvDefine(128, 256)
        self.ec4 = AdaptivePConvDefine(256, 512)

        for i in range(5, 8):
            name = 'ec{:d}'.format(i)
            setattr(self, name, AdaptivePConvDefine(512, 512))
        
        self.OppoMaskConv1 = OppoMaskConv(3, 64, 4, 2, 1, convBias=False, maskInit=0.04)
        self.OppoMaskConv2 = OppoMaskConv(64, 128, 4, 2, 1, convBias=False)
        self.OppoMaskConv3 = OppoMaskConv(128, 256, 4, 2, 1, convBias=False)
        self.OppoMaskConv4 = OppoMaskConv(256, 512, 4, 2, 1, convBias=False)
        self.OppoMaskConv5 = OppoMaskConv(512, 512, 4, 2, 1, convBias=False)
        self.OppoMaskConv6 = OppoMaskConv(512, 512, 4, 2, 1, convBias=False)
 
        self.dc1 = UpAdaptiveConv(512, 512)
        self.dc2 = UpAdaptiveConv(512 * 2, 512)
        self.dc3 = UpAdaptiveConv(512 * 2, 512)
        self.dc4 = UpAdaptiveConv(512 * 2, 256)
        self.dc5 = UpAdaptiveConv(256 * 2, 128)
        self.dc6 = UpAdaptiveConv(128 * 2, 64)
        self.dc7 = nn.ConvTranspose2d(64 * 2, outputChannels, kernel_size=4, stride=2, padding=1, bias=False)

        """ self.lastAdaPConv = nn.ConvTranspose2d(64 * 2, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.lastAdaPActiv = nn.LeakyReLU(0.2)   """      

        self.tanh = nn.Tanh()
    
    def forward(self, inputImgs, masks):
        forCat1, ef1, ms1 = self.ec1(inputImgs, masks)
        forCat2, ef2, ms2 = self.ec2(ef1, ms1)
        forCat3, ef3, ms3 = self.ec3(ef2, ms2)
        forCat4, ef4, ms4 = self.ec4(ef3, ms3)
        forCat5, ef5, ms5 = self.ec5(ef4, ms4)
        forCat6, ef6, ms6 = self.ec6(ef5, ms5)
        _, ef7, _ = self.ec7(ef6, ms6)

        msum1 = self.OppoMaskConv1(1 - masks)
        msum2 = self.OppoMaskConv2(msum1)
        msum3 = self.OppoMaskConv3(msum2)
        msum4 = self.OppoMaskConv4(msum3)
        msum5 = self.OppoMaskConv5(msum4)
        msum6 = self.OppoMaskConv6(msum5)

        catmf1 = torch.cat((ms6, msum6), 1)
        dcf1 = self.dc1(forCat6, ef7, catmf1)
        catmf2 = torch.cat((ms5, msum5), 1)
        dcf2 = self.dc2(forCat5, dcf1, catmf2)

        catmf3 = torch.cat((ms4, msum4), 1)
        dcf3 = self.dc3(forCat4, dcf2, catmf3)

        catmf4 = torch.cat((ms3, msum3), 1)
        dcf4 = self.dc4(forCat3, dcf3, catmf4)

        catmf5 = torch.cat((ms2, msum2), 1)
        dcf5 = self.dc5(forCat2, dcf4, catmf5)

        catmf6 = torch.cat((ms1, msum1), 1)
        dcf6 = self.dc6(forCat1, dcf5, catmf6)
        
        #modified part (adding last adaptive attention)
        """ catmf7 = self.lastAdaPConv(catmf6)
        catmf7 = self.lastAdaPActiv(catmf7) """

        dcf7 = self.dc7(dcf6)
        """ dcf7out = dcf7 * catmf7 """
        #end modified
        output = self.tanh(dcf7)
        output = torch.abs(output)

        return output


##discriminator
#define two column discriminator
class DiscriminatorDoubleColumn(nn.Module):
    def __init__(self, inputChannels):
        super(DiscriminatorDoubleColumn, self).__init__()

        self.globalConv = nn.Sequential(
            nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2 , inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
        )

        self.localConv = nn.Sequential(
            nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2 , inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fusionLayer = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, batches, masks):
        globalFt = self.globalConv(batches * masks)
        localFt = self.localConv(batches * (1 - masks))

        concatFt = torch.cat((globalFt, localFt), 1)

        return self.fusionLayer(concatFt).view(batches.size()[0], -1)

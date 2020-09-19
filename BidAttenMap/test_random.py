import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import utils
from BidAttenMap.dataloader import getDataRandomMask, getDataCombinedMask, getRenxiang, CombinedMaskClustered
#from newModels import CombinedPConvUNet
from BidAttenMap.attentionModels import CombinedNet
import BidAttenMap.pytorch_ssim
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    cudnn.benchmark = True


def bid(dataRoot=None, maskRoot=None, saveRoot=None, Size=None):

    toTensorTrans = Compose([
        ToPILImage(),
        Resize(size=(1500, 2000), interpolation=Image.LINEAR),
        ToTensor()
    ])


    batchSize = 1
    loadSize = Size
    cropSize = Size


    transform = Compose([
            Resize(size=cropSize, interpolation=Image.NEAREST),
            ToTensor(),
        ])

    toTensorTrans = Compose([
        ToPILImage(),
        Resize(size=(1500, 2000), interpolation=Image.LINEAR),
        ToTensor()
    ])

    resultTrans = Compose([
        ToTensor(),
    ])

    dataRoot = dataRoot
    maskRoot = maskRoot
    SavePath = saveRoot
    os.makedirs(SavePath, exist_ok=True)
    #print(maskRoot)
    imgData = CombinedMaskClustered(dataRoot, maskRoot, loadSize, cropSize)
    data_loader = DataLoader(imgData, batch_size=batchSize, shuffle=False, num_workers=1, drop_last=False)

    num_epochs = 10

    netG = CombinedNet(4, 3)
    """ netD = Discriminator() """
    #netD.apply(weights_init())

    #
    if cuda:
        netG = netG.cuda()

        netG.load_state_dict(torch.load('/home/r/shangwei/defence/BidAttenMap/Places160.pth'))
    #/home/r/others/BidAttenMap/BidAttenMap/model/
        #netD.load_state_dict(torch.load('./checkpoints/PartialUNet/Disc_1025.pth'))

    for param in netG.parameters():
        param.requires_grad = False

    print('OK!')

    """ import re
    from matplotlib import pyplot as plt
    import matplotlib.mlab as mlab
    import numpy as np
    cuda = torch.cuda.is_available()
    if cuda:
        print('Cuda is available!')
        cudnn.benchmark = True
    
    
    count = 0
    match = 'MaskConv.weight'
    maskParams = {}
    params = list(netG.named_parameters())
    for name, param in params:
        if re.search(match, name):
            print(count, ' ', name)
            maskParams[count] = param
            count += 1
    
    for i in range(count):
        print(maskParams[i].size())
    
    maskconvs = maskParams[0].data.cpu().detach().numpy()
    print(maskconvs.shape)
    maskconvs = maskconvs.flatten()
    print(maskconvs.shape)
    
    plt.hist(maskconvs, bins=50, edgecolor='k') 
    plt.title("histogram") 
    plt.show()
    
    x1 = np.linspace(maskconvs.min(), maskconvs.max())
    normal = mlab.normpdf(x1, maskconvs.mean(), maskconvs.std())
    line1, = plt.plot(x1,normal,'r-', linewidth = 2)
    plt.show() """

    sum_psnr = 0
    sum_ssim = 0
    count = 0
    sum_time = 0.0

    import time
    # for i in range(1, num_epochs + 1):
    #     netG.eval()
    #     if count >= 20:
    #         break
    netG.eval()
    for inputImgs, GT, masks in (data_loader):

        start = time.time()
        if cuda:
            inputImgs = inputImgs.cuda()
            GT = GT.cuda()
            masks = masks.cuda()
        #long running
        #do something other
        fake_images = netG(inputImgs, masks)
        end = time.time()
        sum_time += (end - start) / batchSize
        print((end - start) / batchSize)
        g_image = fake_images.data.cpu()
        GT = GT.data.cpu()
        mask = masks.data.cpu()
        damaged = GT * mask
        generaredImage = GT * mask + g_image * (1 - mask)
        groundTruth = GT
        masksT = mask
        generaredImage.add(1)
        generaredImage.mul(0.5)
        groundTruth.add(1)
        groundTruth.mul(0.5)
        count += 1
        # batch_mse = ((groundTruth - generaredImage) ** 2).mean()
        # psnr = 10 * math.log10(1 / batch_mse)
        # sum_psnr += psnr
        # print(count, ' psnr:', psnr)
        # ssim = pytorch_ssim.ssim(groundTruth, generaredImage)
        # sum_ssim += ssim
        # print(count, ' ssim:', ssim)
        outputs =torch.Tensor(4 * GT.size()[0], GT.size()[1], cropSize[0], cropSize[1])
        for i in range(GT.size()[0]):
            outputs[4 * i] = masksT[i]
            outputs[4 * i + 1] = damaged[i]
            outputs[4 * i + 2] = generaredImage[i]
            outputs[4 * i + 3] = GT[i]
            #outputs[5 * i + 4] = 1 - masksT[i]
        #save_image(outputs, savePath + 'results-{}'.format(count) + '.png')
        damaged = GT * mask + (1 - mask)

        for j in range(GT.size()[0]):
            #save_image(outputs[4 * j + 1], savePath + '/damaged/damaged{}-{}.png'.format(count, j))
            outputs[4 * j + 1] = damaged[j]

        for j in range(GT.size()[0]):
            outputs[4 * j] = 1- masksT[j]
            # save_image(outputs[4 * j], savePath + '/masks/mask{}-{}.png'.format(count, j))
            # save_image(outputs[4 * j + 1], savePath + '/input/input{}-{}.png'.format(count, j))
            save_image(outputs[4 * j + 2], SavePath + '{}.png'.format(count))
            # save_image(outputs[4 * j + 3], savePath + '/GT/GT{}-{}.png'.format(count, j))








    # print('average psnr:', sum_psnr / count)
    # print('average ssim:', sum_ssim / count)
    print('average time cost:', sum_time / count)

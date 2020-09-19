from os import listdir
from os import walk
from os.path import join
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, Resize, RandomHorizontalFlip
from random import randint
import torchvision.utils as vutils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp', '.BMP'])

def ImageTransform(loadSize, cropSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
        RandomCrop(size=cropSize),
        #RandomHorizontalFlip(p=0.5),
        ToTensor(),
    ])

def MaskTransform(cropSize):
    return Compose([
        Resize(size=cropSize, interpolation=Image.NEAREST),
        ToTensor(),
    ])

class getDataRandomMask(Dataset):
    def __init__(self, dataRoot, maskRoot, loadSize, cropSize):
        super().__init__()
        self.imageFiles = [join (dataRoot, files) for files in listdir(dataRoot) if is_image_file(files)]
        self.masks = [join (maskRoot, mask) for mask in listdir(maskRoot) if is_image_file(mask)]
        self.numberOfMasks = len(self.masks)
        self.loadSize = loadSize
        self.cropSize = cropSize
        self.ImgTrans = ImageTransform(loadSize, cropSize)
        self.maskTrans = MaskTransform(cropSize)
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        mask = Image.open(self.masks[randint(0, self.numberOfMasks - 1)])
        
        groundTruth = self.ImgTrans(img)
        mask = self.maskTrans(mask.convert('RGB'))
        groundTruth.mul(2)
        groundTruth.add(-1)

        inputImage = groundTruth * mask


        return inputImage, groundTruth, mask
    
    def __len__(self):
        return len(self.imageFiles)


class getDataCombinedMask(Dataset):
    def __init__(self, dataRoot, maskRoot, loadSize, cropSize):
        super(getDataCombinedMask, self).__init__()
        self.imageFiles = [join (dataRoot, files) for files in listdir(dataRoot) if is_image_file(files)]
        self.masks = [join (maskRoot, mask) for mask in listdir(maskRoot) if is_image_file(mask)]
        self.numberOfMasks = len(self.masks)
        self.loadSize = loadSize
        self.cropSize = cropSize
        self.ImgTrans = ImageTransform(loadSize, cropSize)
        self.maskTrans = MaskTransform(cropSize)
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        mask = Image.open(self.masks[randint(0, self.numberOfMasks - 1)])
        
        groundTruth = self.ImgTrans(img.convert('RGB'))
        mask = self.maskTrans(mask.convert('RGB'))
        
        inputImage = groundTruth * mask
        inputImage = torch.cat((inputImage, mask[0].view(1, self.cropSize[0], self.cropSize[1])), 0)

        return inputImage, groundTruth, mask
    
    def __len__(self):
        return len(self.imageFiles)



class getDataWithCombinedAndMean(Dataset):
    def __init__(self, dataRoot, maskRoot, loadSize, cropSize):
        super().__init__()
        self.imageFiles = [join (dataRoot, files) for files in listdir(dataRoot) if is_image_file(files)]
        self.masks = [join (maskRoot, files) for files in listdir(maskRoot) if is_image_file(files)]
        self.numberOfMasks = len(self.masks)
        self.cropSize = cropSize
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform(loadSize, cropSize)
        self.maskTrans = MaskTransform(cropSize)

    def __len__(self):
        return len(self.imageFiles)
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        mask = Image.open(self.masks[randint(0, self.numberOfMasks - 1)])

        groundTruth = self.ImgTrans(img)
        mask = self.maskTrans(mask.convert('RGB'))
        groundTruth.mul(2)
        groundTruth.add(-1)

        inputImgs = groundTruth * mask
        holes = groundTruth == 0
        validSum = torch.sum(mask[0])
        numPixels = self.cropSize ** 2
        inputImgs[0].masked_fill_(holes[0], 0.1 * inputImgs[0].mean() * (256 ** 2) / (numPixels - validSum))
        inputImgs[1].masked_fill_(holes[1], 0.1 * inputImgs[1].mean() * (256 ** 2) / (numPixels - validSum))
        inputImgs[2].masked_fill_(holes[2], 0.1 * inputImgs[2].mean() * (256 ** 2) / (numPixels -validSum))

        inputImgs = torch.cat((inputImgs, mask[0].view(1, self.cropSize, self.cropSize)), 0)

        return inputImgs, groundTruth, mask


class CombinedMaskClustered(Dataset):
    def __init__(self, dataRoot, maskRoot, loadSize, cropSize):
        super(CombinedMaskClustered, self).__init__()
        #self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
        #    for files in filenames if is_image_file(files)]
        self.imageFiles = sorted([join(dataRoot, files) for files in listdir(dataRoot) if is_image_file(files)])
        self.masks = sorted([join(maskRoot, mask) for mask in listdir(maskRoot) if is_image_file(mask)])
        #self.masks = [join (maskRootK, files) for maskRootK, dn, filenames in walk(maskRoot) \
        #    for files in filenames if is_image_file(files)]
        self.numberOfMasks = len(self.masks)
        self.loadSize = loadSize
        self.cropSize = cropSize
        self.ImgTrans = ImageTransform(loadSize, cropSize)
        self.maskTrans = MaskTransform(cropSize)
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        # basename = self.imageFiles[index]
        #
        # # basename1 = basename.split('.')[0]
        # # num = basename1[-1]
        # # basename2 = self.masks[index]
        # # basename3 = basename2[:43] + num + basename2[44:]
        # basename3 = basename[:48] + 'labels' + basename[54:70] + '.png'
        
        mask = Image.open(self.masks[index])
        
        groundTruth = self.ImgTrans(img.convert('RGB'))
        mask = self.maskTrans(mask.convert('RGB'))
        """ groundTruth  = groundTruth.mul(2)
        groundTruth = groundaTruth.add(-1)
 """
        inputImage = groundTruth * mask
        inputImage = torch.cat((inputImage, mask[0].view(1, self.cropSize[0], self.cropSize[1])), 0)

        return inputImage, groundTruth, mask
    
    def __len__(self):
        return len(self.imageFiles)


class getRenxiang(Dataset):
    def __init__(self, dataRoot, maskRoot, loadSize, cropSize):
        super(getRenxiang, self).__init__()
        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if is_image_file(files)]
        self.masksRoot = maskRoot
        self.loadSize = loadSize
        self.cropSize = cropSize
        self.ImgTrans = ImageTransform(loadSize, cropSize)
        self.maskTrans = MaskTransform(cropSize)
    
    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        imgName = self.imageFiles[index].split('/')[-1]
        numberName = img.splie('_')[-1]
        mask = Image.open(self.masksRoot + '/masks/'+ 'mask_' + numberName)
        maskDila = Image.open(self.masksRoot + '/dilationMask/' + 'mask_' + numberName)
        mask = self.maskTrans(mask.convert('RGB'))
        maskDila = self.maskTrans(maskDila.convert('RGB'))

        groundTruth = self.ImgTrans(img)

        inputImage = groundTruth * maskDila + groundTruth * mask
        maskInput = 0

        inputImage = torch.cat((inputImage, mask[0].view(1, self.cropSize[0], self.cropSize[1])), 0)
        
        return inputImage, groundTruth, mask

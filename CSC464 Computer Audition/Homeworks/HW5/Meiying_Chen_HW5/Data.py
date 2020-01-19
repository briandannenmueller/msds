# @author Yujia Yan

from __future__ import print_function, division, absolute_import, unicode_literals
import six

import os

import numpy as np

import util
import sys
import torch
import torchvision.transforms as transforms 
import torch.utils.data


class DSD100Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split = 'Train', mono = True, transform = None, repetition = 1):
        mixturePaths = []
        vocalPaths = []

        mixturePath = os.path.join(path, 'Mixtures')
        sourcePath = os.path.join(path, 'Sources')
        
        mixturePath = os.path.join(mixturePath, split)
        sourcePath = os.path.join(sourcePath, split)


        for aFile in os.listdir(mixturePath):
            aFileMixture = os.path.join(mixturePath,aFile)
            aFileSource = os.path.join(sourcePath, aFile)
            if os.path.isdir(aFileMixture) and os.path.isdir(aFileSource):
                wavFilePath = os.path.join(aFileMixture,'mixture.wav')
                vocalFilePath = os.path.join(aFileSource, 'vocals.wav')

                if(os.path.isfile(wavFilePath) and vocalFilePath):
                    mixturePaths.append(wavFilePath)
                    vocalPaths.append(vocalFilePath)

        self.mixturePaths = mixturePaths*repetition
        self.vocalPaths = vocalPaths*repetition
        self.split = split
        self.mono = mono
        self.transform = transform
    def __len__(self):
        return len(self.mixturePaths)

    def __getitem__(self, idx):
        if idx >= len(self.mixturePaths):
            raise IndexError
        mixture, fs = util.wavread(self.mixturePaths[idx])
        vocal, fs= util.wavread(self.vocalPaths[idx])
        if self.mono:
            #downmix here
            mixture = np.mean(mixture, axis=  -1)
            vocal= np.mean(vocal, axis=  -1)
        
        sample = {'mixture':mixture.astype(np.float32), 'vocal':vocal.astype(np.float32)}
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

class Transforms:
    class Rescale:
        def __init__(self, minFactor, maxFactor):
            self.minFactor = minFactor 
            self.maxFactor = maxFactor

        def __call__(self, sample):
            # shift = np.random.randint(self.maxShift)
            factor = np.random.rand()* (self.maxFactor - self.minFactor)+ self.minFactor
            return {key: value*factor for key,value in sample.items()}

    class RandomShift:
        def __init__(self, maxShift):
            self.maxShift = maxShift

        def __call__(self, sample):
            shift = np.random.randint(self.maxShift)
            return {key:np.pad(value, [(shift, 0)] + [(0,0)]* (len(value.shape)-1), mode='constant') for key,value in sample.items()}

    class MakeMagnitudeSpectrum:
        def __init__(self, blockSize, hopSize, window = "hamming"):
            self.blockSize = blockSize
            self.hopSize = hopSize
            self.window = window

        def __call__(self, sample):
            
            aa = {key:np.abs( util.stft_real(value,
                        blockSize = self.blockSize,
                        hopSize = self.hopSize,
                        window = self.window)) for key, value in sample.items() }
            
            
            return aa

    
    class ShuffleFrameOrder:
        def __init__(self):
            pass
        
        def __call__(self, sample):
            nFrame = sample["mixture"].shape[1]
            # print(nFrame)
            order = np.arange(nFrame)
            np.random.shuffle(order)

            # print({key: value[:,order] for key,value in sample.items()})
            return {key: value[:,order] for key,value in sample.items()}


def collate_fn(batch):
    nBatch = len(batch)
    resultBatch = {}
    for key in batch[0]:
        maxLen = batch[0][key].shape[1]
        nWindow = batch[0][key].shape[0]
        for i in range(nBatch):
            maxLen = max(batch[i][key].shape[1], maxLen)

        tmp = torch.zeros(nBatch, nWindow, maxLen)
        for i in range(nBatch):
            tmp[i, :, :batch[i][key].shape[1]] = torch.from_numpy(batch[i][key])
        
        resultBatch[key]= tmp
    return resultBatch


if __name__ == "__main__":
    transform = transforms.Compose([
        Transforms.RandomShift(2048),
        Transforms.MakeMagnitudeSpectrum(blockSize=4096, hopSize = 2048)])
    path_to_dsd100 = "../DSD100/"
    dataset = DSD100Dataset(path_to_dsd100, split = 'Valid', mono =True, transform = transform)

    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1,shuffle=True, num_workers = 0)
    dataloaderIt = iter(dataloader)
    aa = next(dataloaderIt) # TODO
    for idx, sample in enumerate(dataloader):
        print(sample['vocal'].shape)
        print(sample['mixture'].shape)


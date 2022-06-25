#!/usr/bin/env python

import cv2
import os
import sys
import ast
import torch
import numpy as np
import math
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)

def aug_brightness(npyIn, fltBrightness):
    assert(type(npyIn) == np.ndarray or type(npyIn[0]) == np.ndarray)
    assert(npyIn.ndim == 3 if type(npyIn) == np.ndarray else npyIn[0].ndim == 3)
    assert(npyIn.shape[2] == 3 if type(npyIn) == np.ndarray else npyIn[0].shape[2] == 3)
    assert(npyIn.dtype in [np.uint8] if type(npyIn) == np.ndarray else npyIn[0].dtype in [np.uint8])
    assert(type(fltBrightness) == float and fltBrightness >= 0.0 and fltBrightness <= 2.0)

    npyBrightness = np.array([intColor * fltBrightness for intColor in range(256)]).clip(0.0, 255.0).astype(np.uint8)

    npyOuts = []

    for npyOut in [npyIn] if type(npyIn) == np.ndarray else npyIn:
        npyOuts.append(cv2.LUT(src=npyOut, lut=npyBrightness))
    # end

    return npyOuts[0] if type(npyIn) == np.ndarray else npyOuts
# end


def aug_contrast(npyIn, fltContrast):
    assert(type(npyIn) == np.ndarray or type(npyIn[0]) == np.ndarray)
    assert(npyIn.ndim == 3 if type(npyIn) == np.ndarray else npyIn[0].ndim == 3)
    assert(npyIn.shape[2] == 3 if type(npyIn) == np.ndarray else npyIn[0].shape[2] == 3)
    assert(npyIn.dtype in [np.uint8] if type(npyIn) == np.ndarray else npyIn[0].dtype in [np.uint8])
    assert(type(fltContrast) == float and fltContrast >= 0.0 and fltContrast <= 2.0)

    npyContrast = []

    for npyOut in [npyIn] if type(npyIn) == np.ndarray else npyIn:
        npyContrast.append(npyOut.mean())
    # end

    npyContrast = sum(npyContrast) / len(npyContrast)

    npyContrast = np.array([((intColor - npyContrast) * fltContrast) + npyContrast for intColor in range(256)]).clip(0.0, 255.0).astype(np.uint8)

    npyOuts = []

    for npyOut in [npyIn] if type(npyIn) == np.ndarray else npyIn:
        npyOuts.append(cv2.LUT(src=npyOut, lut=npyContrast))
    # end

    return npyOuts[0] if type(npyIn) == np.ndarray else npyOuts
# end


def aug_hue(npyIn, fltHue):
    assert(type(npyIn) == np.ndarray or type(npyIn[0]) == np.ndarray)
    assert(npyIn.ndim == 3 if type(npyIn) == np.ndarray else npyIn[0].ndim == 3)
    assert(npyIn.shape[2] == 3 if type(npyIn) == np.ndarray else npyIn[0].shape[2] == 3)
    assert(npyIn.dtype in [np.uint8] if type(npyIn) == np.ndarray else npyIn[0].dtype in [np.uint8])
    assert(type(fltHue) == float and fltHue >= -0.5 and fltHue <= 0.5)

    npyHue = (np.array([intColor + (180.0 * fltHue) for intColor in range(256)]) % 180.0).astype(np.uint8)

    npyOuts = []

    for npyOut in [npyIn] if type(npyIn) == np.ndarray else npyIn:
        npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_BGR2HSV)

        npyOut[:, :, 0] = cv2.LUT(src=npyOut[:, :, 0], lut=npyHue)

        npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_HSV2BGR)

        npyOuts.append(npyOut)
    # end

    return npyOuts[0] if type(npyIn) == np.ndarray else npyOuts
# end


def aug_saturation(npyIn, fltSaturation):
    assert(type(npyIn) == np.ndarray or type(npyIn[0]) == np.ndarray)
    assert(npyIn.ndim == 3 if type(npyIn) == np.ndarray else npyIn[0].ndim == 3)
    assert(npyIn.shape[2] == 3 if type(npyIn) == np.ndarray else npyIn[0].shape[2] == 3)
    assert(npyIn.dtype in [np.uint8] if type(npyIn) == np.ndarray else npyIn[0].dtype in [np.uint8])
    assert(type(fltSaturation) == float and fltSaturation >= 0.0 and fltSaturation <= 2.0)

    npyOuts = []

    for npyOut in [npyIn] if type(npyIn) == np.ndarray else npyIn:
        npyOuts.append(cv2.addWeighted(npyOut, fltSaturation, cv2.cvtColor(src=cv2.cvtColor(src=npyOut, code=cv2.COLOR_BGR2GRAY), code=cv2.COLOR_GRAY2BGR), 1.0 - fltSaturation, 0))
    # end

    return npyOuts[0] if type(npyIn) == np.ndarray else npyOuts
# end


def aug_equalize(npyIn, strType, strColor):
    assert(type(npyIn) == np.ndarray or type(npyIn[0]) == np.ndarray)
    assert(npyIn.ndim == 3 if type(npyIn) == np.ndarray else npyIn[0].ndim == 3)
    assert(npyIn.shape[2] == 3 if type(npyIn) == np.ndarray else npyIn[0].shape[2] == 3)
    assert(npyIn.dtype in [np.uint8] if type(npyIn) == np.ndarray else npyIn[0].dtype in [np.uint8])
    assert(type(strType) == str and strType.split('-')[0] in ['global', 'local'])
    assert(type(strColor) == str and strColor in ['all', 'yuv', 'lab'])

    npyOuts = []

    for npyOut in [npyIn] if type(npyIn) == np.ndarray else npyIn:
        if strType.split('-')[0] == 'global' and strColor == 'all':
            npyOut = np.stack([cv2.equalizeHist(src=npyOut[:, :, intChan]) for intChan in range(3)], 2)

        elif strType.split('-')[0] == 'global' and strColor == 'yuv':
            npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_BGR2YUV)
            npyOut[:, :, 0] = cv2.equalizeHist(src=npyOut[:, :, 0])
            npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_YUV2BGR)

        elif strType.split('-')[0] == 'global' and strColor == 'lab':
            npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_BGR2Lab)
            npyOut[:, :, 0] = cv2.equalizeHist(src=npyOut[:, :, 0])
            npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_Lab2BGR)

        elif strType.split('-')[0] == 'local' and strColor == 'all':
            npyOut = np.stack([cv2.createCLAHE(clipLimit=float(strType.split('-')[1]), tileGridSize=(8, 8)).apply(src=npyOut[:, :, intChan]) for intChan in range(3)], 2) if float(strType.split('-')[1]) != 0.0 else npyOut

        elif strType.split('-')[0] == 'local' and strColor == 'yuv':
            npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_BGR2YUV)
            npyOut[:, :, 0] = cv2.createCLAHE(clipLimit=float(strType.split('-')[1]), tileGridSize=(8, 8)).apply(src=npyOut[:, :, 0]) if float(strType.split('-')[1]) != 0.0 else npyOut[:, :, 0]
            npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_YUV2BGR)

        elif strType.split('-')[0] == 'local' and strColor == 'lab':
            npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_BGR2Lab)
            npyOut[:, :, 0] = cv2.createCLAHE(clipLimit=float(strType.split('-')[1]), tileGridSize=(8, 8)).apply(src=npyOut[:, :, 0]) if float(strType.split('-')[1]) != 0.0 else npyOut[:, :, 0]
            npyOut = cv2.cvtColor(src=npyOut, code=cv2.COLOR_Lab2BGR)

        # end

        npyOuts.append(npyOut)
    # end

    return npyOuts[0] if type(npyIn) == np.ndarray else npyOuts
# end


class VimeoDataset(Dataset):
    def __init__(self, dataset_split, dataset_path='./dataset/vimeo_triplet/', batch_size=32):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.datalist = self.load_data(dataset_split)
        
        self.h = 256
        self.w = 448
        print("Found %d images for mode: %s" % (self.__len__(),dataset_split))
    # end


    def __len__(self):
        return len(self.datalist)
    # end


    def load_data(self,dataset_split):
        list_ = []
        if dataset_split=='train':
            list_name = 'tri_trainlist.txt'
        elif dataset_split=='validation':
            list_name = 'tri_testlist.txt'
        else:
            raiseNotImplementedError("only Train or Val mode for Vimeo90!")
        # end

        for line in open(os.path.join(self.dataset_path,list_name)): 
            if not os.path.exists(os.path.join(self.dataset_path,'sequences',line.strip('\n'),'im1.png')):
                print(line) 
                pdb.set_trace()            
            # end            
            list_.append(line.strip('\n'))
        # end
        return list_
    # end


    def aug(self, img0, gt, img1, h, w):
        
        ih, iw, _ = img0.shape

        #Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]

        #Flip Color
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, :, ::-1]
            img1 = img1[:, :, ::-1]
            gt = gt[:, :, ::-1]
        # end

        #Flip Vertical
        if random.uniform(0, 1) < 0.5:
            img0 = img0[::-1]
            img1 = img1[::-1]
            gt = gt[::-1]
        # end

        #Flip Horizontal
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, ::-1]
            img1 = img1[:, ::-1]
            gt = gt[:, ::-1]

        #Flip Temporal Order
        if random.uniform(0, 1) < 0.5:
            tmp = img1
            img1 = img0
            img0 = tmp
        # end

        return img0.astype(np.float32), gt.astype(np.float32), img1.astype(np.float32)
    # end


    def augment(self, img0, gt, img1, hs, ws):
        npySeq = [img0,img1,gt]

        npySeq[:-1] = list(reversed(npySeq[:-1])) if random.random() < 0.5 else npySeq[:-1]

        intX = int(round((random.random()) * (npySeq[0].shape[1] - ws)))
        intY = int(round((random.random()) * (npySeq[0].shape[0] - hs)))
        npySeq = [npyFrame[intY:intY + hs, intX:intX + ws, :] for npyFrame in npySeq]

        intVflip = random.choice([1, -1])
        intHflip = random.choice([1, -1])
        intRotate = random.choice([0, 1, 2, 3])
        npySeq = [npyFrame[::intVflip, ::intHflip, :] for npyFrame in npySeq]
        npySeq = [np.rot90(npyFrame, intRotate, [0, 1]) for npyFrame in npySeq]
        npySeq = [np.ascontiguousarray(npyFrame) for npyFrame in npySeq]

        for strType in random.sample(['brightness', 'contrast', 'hue', 'saturation'], 3):
            npySeq = aug_brightness(npySeq, random.uniform(0.9, 1.1)) if strType == 'brightness' else npySeq
            npySeq = aug_contrast(npySeq, random.uniform(0.9, 1.1)) if strType == 'contrast' else npySeq
            npySeq = aug_hue(npySeq, random.uniform(-0.05, 0.05)) if strType == 'hue' else npySeq
            npySeq = aug_saturation(npySeq, random.uniform(0.8, 1.2)) if strType == 'saturation' else npySeq
        # end

        npySeq = aug_equalize(npySeq, random.choice(['global', 'local-' + format(random.uniform(0.0, 1.0), '.9f')]), random.choice(['all', 'yuv', 'lab'])) if random.random() < 0.1 else npySeq

        img0,img1,gt = npySeq

        return img0.astype(np.float32), gt.astype(np.float32), img1.astype(np.float32)
    # end


    def getimg(self, index):
        sample = self.datalist[index]
        img0 = np.array(cv2.imread(os.path.join(self.dataset_path,'sequences',sample,'im1.png'))).astype(np.uint8)
        img1 = np.array(cv2.imread(os.path.join(self.dataset_path,'sequences',sample,'im3.png'))).astype(np.uint8)
        gt = np.array(cv2.imread(os.path.join(self.dataset_path,'sequences',sample,'im2.png'))).astype(np.uint8)

        if len(img0.shape)==2:
            print(os.path.join(self.dataset_path,'sequences',sample,'im1.png'))
            img0 = img0[:,:, np.newaxis]
            img0 = img0.repeat(3, axis=2)
        # end

        if len(img1.shape)==2:
            img1 = img1[:,:, np.newaxis]
            img1 = img1.repeat(3, axis=2)
        # end

        if len(gt.shape)==2:
            gt = gt[:,:, np.newaxis]
            gt = gt.repeat(3, axis=2)
        # end

        return img0, gt, img1
    # end


    def __getitem__(self, index):        
        img0, gt, img1 = self.getimg(index)
        
        if self.dataset_split == 'train':
            img0, gt, img1 = self.augment(img0, gt, img1, 256, 256)
        # end

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        time_loc = torch.tensor(0.5).view(1,1,1).float()

        return img0, img1, gt, time_loc
    # end
# end


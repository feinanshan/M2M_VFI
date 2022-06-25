#!/usr/bin/env python

import glob
import numpy
import os
import cv2
import math
import PIL.Image
import torch
import tqdm

from model.pytorch_msssim import  ssim_matlab
import model.m2m as m2m


##########################################################
torch.set_grad_enabled(False) 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

netNetwork = m2m.M2M_PWC().cuda().eval()

netNetwork.load_state_dict(torch.load('./model_best.pkl'))


strPath = '/PATH/TO/atd12k/test_2k_540p/'

##########################################################

if __name__ == '__main__':
    fltPsnr = []
    fltSsim = []

    lstDirs = os.listdir(strPath)

    for strDir in tqdm.tqdm(lstDirs):
        npyOne = numpy.array(PIL.Image.open(strPath + strDir + '/frame1.png'))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)
        npyTwo = numpy.array(PIL.Image.open(strPath + strDir + '/frame3.png'))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)
        npyTruth = numpy.array(PIL.Image.open(strPath + strDir + '/frame2.png'))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)


        tenOne = torch.FloatTensor(numpy.ascontiguousarray(npyOne.transpose(2, 0, 1)[None, :, :, :])).cuda()
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(npyTwo.transpose(2, 0, 1)[None, :, :, :])).cuda()
        tenGT = torch.FloatTensor(numpy.ascontiguousarray(npyTruth.transpose(2, 0, 1)[None, :, :, :])).cuda()

        tenEstimate = netNetwork(tenOne, tenTwo, [torch.FloatTensor([0.5]).view(1, 1, 1, 1).cuda()])[0]
        npyEstimate = (tenEstimate.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).round().astype(numpy.uint8)
        tenEstimate = torch.FloatTensor(npyEstimate.transpose(2, 0, 1)[None, :, :, :]).cuda() / 255.0

        fltPsnr.append(-10 * math.log10(torch.mean((tenEstimate - tenGT) * (tenEstimate - tenGT)).cpu().data))
        fltSsim.append(ssim_matlab(tenEstimate,tenGT).detach().cpu().numpy())
    # end

    print('computed average psnr', numpy.mean(fltPsnr))
    print('computed average ssim', numpy.mean(fltSsim))
# end

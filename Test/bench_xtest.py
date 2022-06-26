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



def getXVFI(dir, multiple=8, t_step_size=32):
    """ make [I0,I1,It,t,scene_folder] """
    testPath = []
    t = numpy.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
    for type_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):  # [type1,type2,type3,...]
        for scene_folder in sorted(glob.glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
            frame_folder = sorted(glob.glob(scene_folder + '*.png'))  # 32 multiple, ['00000.png',...,'00032.png']
            for idx in range(0, len(frame_folder), t_step_size):  # 0,32,64,...
                if idx == len(frame_folder) - 1:
                    break
                #end
                for mul in range(multiple - 1):
                    I0I1It_paths = []
                    I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
                    I0I1It_paths.append(frame_folder[idx + t_step_size])  # I1 (fix)
                    I0I1It_paths.append(frame_folder[idx + int((t_step_size // multiple) * (mul + 1))])  # It
                    I0I1It_paths.append(t[mul])
                    testPath.append(I0I1It_paths)
                #end

            #end

        #end

    #end
    return testPath
#end




##########################################################
torch.set_grad_enabled(False) 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

netNetwork = m2m.M2M_PWC().cuda().eval()

netNetwork.load_state_dict(torch.load('./model.pkl'))

strPath = '/PATH/TO/XVFI/test/'

##########################################################


if __name__ == '__main__':
    fltPsnr = []
    fltSsim = []

    listFiles = getXVFI(strPath)

    for strMode in ['XTEST-2k', 'XTEST-4k']:
        for intFrame in tqdm.tqdm(listFiles):  
            npyOne = numpy.array(PIL.Image.open(intFrame[0]))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)
            npyTwo = numpy.array(PIL.Image.open(intFrame[1]))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)
            npyTruth = numpy.array(PIL.Image.open(intFrame[2]))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)

            if strMode == 'XTEST-2k': #downsample
                npyOne = cv2.resize(src=npyOne, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                npyTwo = cv2.resize(src=npyTwo, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                npyTruth = cv2.resize(src=npyTruth, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                intRatio=8
            elif strMode == 'XTEST-4k': #downsample
                intRatio=16                    
            #end

            tenOne = torch.FloatTensor(numpy.ascontiguousarray(npyOne.transpose(2, 0, 1)[None, :, :, :])).cuda()
            tenTwo = torch.FloatTensor(numpy.ascontiguousarray(npyTwo.transpose(2, 0, 1)[None, :, :, :])).cuda()
            tenGT = torch.FloatTensor(numpy.ascontiguousarray(npyTruth.transpose(2, 0, 1)[None, :, :, :])).cuda()
            tenTime = torch.tensor(intFrame[3]).view(1,1,1,1).cuda()

            tenEstimate = netNetwork(tenOne, tenTwo, [tenTime], intRatio)[0]
            npyEstimate = (tenEstimate.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).round().astype(numpy.uint8)
            tenEstimate = torch.FloatTensor(npyEstimate.transpose(2, 0, 1)[None, :, :, :]).cuda() / 255.0

            fltPsnr.append(-10 * math.log10(torch.mean((tenEstimate - tenGT) * (tenEstimate - tenGT)).cpu().data))
            fltSsim.append(ssim_matlab(tenEstimate,tenGT).detach().cpu().numpy())

        #end
        print('computed average psnr for',strMode, numpy.mean(fltPsnr))
        print('computed average ssim for',strMode, numpy.mean(fltSsim))
    # end
#end

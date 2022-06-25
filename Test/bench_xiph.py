#!/usr/bin/env python

import glob
import os
import sys
import cv2
import numpy
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


strPath = '/PATH/TO/XiPH/netflix/'

##########################################################


##########################################################
if False:  #prepate data https://github.com/sniklaus/softmax-splatting/blob/master/benchmark.py
    print('this benchmark script can be used to compute the Xiph metrics from our paper')
    print('please note that it uses the SepConv method for doing the actual interpolation')
    print('be aware that the script first downloads about 12 gigabytes of data from Xiph')
    print('do you want to continue with the execution of this script? [y/n]')

    if input().lower() != 'y':
        sys.exit(0)
    # end

    ##########################################################

    if os.path.exists('./netflix') == False:
        os.makedirs('./netflix')
    # end

    if len(glob.glob('./netflix/BoxingPractice-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_BoxingPractice_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/BoxingPractice-%03d.png')
    # end

    if len(glob.glob('./netflix/Crosswalk-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/Crosswalk-%03d.png')
    # end

    if len(glob.glob('./netflix/DrivingPOV-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/Chimera/Netflix_DrivingPOV_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/DrivingPOV-%03d.png')
    # end

    if len(glob.glob('./netflix/FoodMarket-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/FoodMarket-%03d.png')
    # end

    if len(glob.glob('./netflix/FoodMarket2-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket2_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/FoodMarket2-%03d.png')
    # end

    if len(glob.glob('./netflix/RitualDance-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_RitualDance_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/RitualDance-%03d.png')
    # end

    if len(glob.glob('./netflix/SquareAndTimelapse-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/SquareAndTimelapse-%03d.png')
    # end

    if len(glob.glob('./netflix/Tango-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/Tango-%03d.png')
    # end
# end
##########################################################





if __name__ == '__main__':

    for strMode in ['Xiph-2k', 'Xiph-4k']:

        fltPsnr = []
        fltSsim = []

        for strFile in ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 'SquareAndTimelapse', 'Tango']:
            for intFrame in tqdm.tqdm(range(2, 99, 2)):
                npyOne = numpy.array(PIL.Image.open(strPath + strFile +  '-' + str(intFrame - 1).zfill(3) + '.png'))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)
                npyTwo = numpy.array(PIL.Image.open(strPath + strFile +  '-' + str(intFrame + 1).zfill(3) + '.png'))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)
                npyTruth = numpy.array(PIL.Image.open(strPath + strFile +  '-' + str(intFrame  ).zfill(3) + '.png'))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)

                if strMode == 'Xiph-2k': #downsample
                    npyOne = cv2.resize(src=npyOne, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                    npyTwo = cv2.resize(src=npyTwo, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                    npyTruth = cv2.resize(src=npyTruth, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                    intRatio=4
                elif strMode == 'Xiph-4k': #center-crop
                    npyOne = npyOne[540:-540, 1024:-1024, :]
                    npyTwo = npyTwo[540:-540, 1024:-1024, :] 
                    npyTruth = npyTruth[540:-540, 1024:-1024, :]
                    intRatio=8
                #end

                tenOne = torch.FloatTensor(numpy.ascontiguousarray(npyOne.transpose(2, 0, 1)[None, :, :, :])).cuda()
                tenTwo = torch.FloatTensor(numpy.ascontiguousarray(npyTwo.transpose(2, 0, 1)[None, :, :, :])).cuda()
                tenGT = torch.FloatTensor(numpy.ascontiguousarray(npyTruth.transpose(2, 0, 1)[None, :, :, :])).cuda()

                tenEstimate = netNetwork(tenOne, tenTwo, [torch.FloatTensor([0.5]).view(1, 1, 1, 1).cuda()],intRatio)[0]
                npyEstimate = (tenEstimate.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).round().astype(numpy.uint8)
                tenEstimate = torch.FloatTensor(npyEstimate.transpose(2, 0, 1)[None, :, :, :]).cuda() / 255.0

                fltPsnr.append(-10 * math.log10(torch.mean((tenEstimate - tenGT) * (tenEstimate - tenGT)).cpu().data))
                fltSsim.append(ssim_matlab(tenEstimate,tenGT).detach().cpu().numpy())
            # end
        # end
        print('computed average psnr for',strMode, numpy.mean(fltPsnr))
        print('computed average ssim for',strMode, numpy.mean(fltSsim))
# end


#!/usr/bin/env python

import torch
import time
import model.m2m as m2m

##########################################################

torch.set_grad_enabled(False) 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

netNetwork = m2m.M2M_PWC().cuda().eval()

intRatio = 8
intStep = 7
lstSize = [4096//2,2160//2]

##########################################################

if __name__ == '__main__':
    tenOne = torch.randn(1,3,lstSize[0],lstSize[1]).cuda()
    tenTwo = torch.randn(1,3,lstSize[0],lstSize[1]).cuda()
    tenSteps = [torch.FloatTensor([st / intStep*1]).view(1, 1, 1, 1).cuda() for  st in range(0,intStep)] 

    for i in range(10):
     netNetwork(tenOne, tenTwo, tenSteps, intRatio)

    torch.cuda.synchronize()
    t_0 = time.time()

    for i in range(50):

        netNetwork(tenOne, tenTwo, tenSteps,intRatio)
        
    torch.cuda.synchronize()
    t_1 = time.time()

    print('size', lstSize, 'steps', intStep, ':', 1000*(t_1-t_0)/(intStep*50),'ms/f')





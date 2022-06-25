#!/usr/bin/env python

import torch
import math
import numpy as np

class Optimizer(object):
    def __init__(self,
                model,
                name='AdamW',
                lr0=1.0e-4,
                wd=0.0,
                max_iter=100000,
                warmup_steps=2000,
                *args, **kwargs):

        self.model = model
        self.warmup_steps = warmup_steps
        self.max_iter = float(max_iter)
        self.it = 0
        self.wd = wd

        param_list = model.get_params(lr0)

        self.initial_lrs = [param_group['lr'] for param_group in param_list]

        if name=='AdamW':
            self.optim = torch.optim.AdamW(
                param_list,
                lr = 1.0e-4,
                weight_decay = wd)
        elif name=='Adam':
            self.optim = torch.optim.Adam(
                param_list,
                lr = 1.0e-4)
        else:
            raise Exception("Only support Adam and AdamW.")
        #end
    #end

    def set_lr(self):
        if self.it < self.warmup_steps:
            mul = self.it / self.warmup_steps
        else:
            mul = np.cos((self.it - self.warmup_steps) / (self.max_iter - self.warmup_steps) * math.pi) * 0.5 + 0.5
        #end

        for param_group,initial_lr in zip(self.optim.param_groups, self.initial_lrs):
            param_group['lr'] = initial_lr*mul
        #end
    #end


    def step(self):
        self.lr = self.set_lr()
        self.optim.step()
        self.it += 1
    #end

    def zero_grad(self):
        self.optim.zero_grad()
    #end
#end

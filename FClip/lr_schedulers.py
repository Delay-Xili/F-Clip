from __future__ import absolute_import
from __future__ import print_function

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


def init_lr_scheduler(optimizer,
                      lr_scheduler='multi_step', # learning rate scheduler
                      stepsize=[20, 40], # step size to decay learning rate
                      gamma=0.1, # learning rate decay
                      max_epoch=240,
                      eta_min=0,
                      warmUp_epoch=5,
                      last_epoch=-1
                      ):
    if lr_scheduler == 'single_step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize[0], gamma=gamma, last_epoch=last_epoch)
    elif lr_scheduler == 'multi_step':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=stepsize, gamma=gamma, last_epoch=last_epoch)
    elif lr_scheduler == 'cos_step':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=eta_min, last_epoch=last_epoch)
    elif lr_scheduler == 'warmUp_step':
        lr_lambda = lambda epoch: min(1.0, epoch/warmUp_epoch)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
    elif lr_scheduler == 'warmUpSingle_step':
        return WarmUpSingle(optimizer, step_size=stepsize[0], gamma=gamma, warmUp_epoch=warmUp_epoch, last_epoch=last_epoch)
    elif lr_scheduler == 'warmUpCos_step':
        return WarmUpCosine(optimizer, T_max=max_epoch, eta_min=eta_min, warmUp_epoch=warmUp_epoch, last_epoch=last_epoch)

    else:
        raise ValueError('Unsupported lr_scheduler: {}'.format(lr_scheduler))


class WarmUpCosine(_LRScheduler):

    def __init__(self, optimizer, T_max, eta_min=0, warmUp_epoch=5, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmUp_epoch = warmUp_epoch
        super(WarmUpCosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = []
        for base_lr in self.base_lrs:
            if self.last_epoch <= self.warmUp_epoch:
                lr.append(base_lr * self.last_epoch / self.warmUp_epoch)
            else:
                lr.append(self.eta_min + (base_lr - self.eta_min) *
                          (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2)
        return lr


class WarmUpSingle(_LRScheduler):

    def __init__(self, optimizer, step_size, gamma=0.1, warmUp_epoch=10, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.warmUp_epoch = warmUp_epoch
        super(WarmUpSingle, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = []
        for base_lr in self.base_lrs:
            if self.last_epoch <= self.warmUp_epoch:
                lr.append(base_lr * self.last_epoch / self.warmUp_epoch)
            else:
                lr.append(base_lr * self.gamma ** (self.last_epoch // self.step_size))
        return lr
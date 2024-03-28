"""
@author: nghiant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

class CoreModuleStage(nn.Module):
    def __init__(self):
        super(CoreModuleStage, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1))

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x):
        z = x
        for i in range(len(self.conv)):
            z = F.relu(self.conv[i](z))
        return z

class CoreModule(nn.Module):
    def __init__(self, ns):
        super(CoreModule, self).__init__()

        self.ns = ns
        
        self.stages = nn.ModuleList()
        for i in range(ns):
            self.stages.append(CoreModuleStage())

    def forward(self, x, s=None):
        if s is not None:
            z = self.stages[s](x)
            return z

        z = x
        for s in range(self.ns):
            z = self.stages[s](z)
        return z

class MaskModuleStage(nn.Module):
    def __init__(self):
        super(MaskModuleStage, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(16, 4, 1, 1, 0))
        self.conv.append(nn.Conv2d(4,  4, 3, 1, 1))
        self.conv.append(nn.Conv2d(4, 16, 1, 1, 0))

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x):
        z = x
        for i in range(len(self.conv) - 1):
            z = F.relu(self.conv[i](z))
        d = self.conv[i+1](z)
        p = torch.sigmoid(d)

        return p

class MaskModule(nn.Module):
    def __init__(self, ns):
        super(MaskModule, self).__init__()

        self.ns = ns

        self.stages = nn.ModuleList()
        for i in range(ns):
            self.stages.append(MaskModuleStage())

    def forward(self, x, s):
        p = self.stages[s](x)
        return p

class _MaskNetB_7_ns(nn.Module): #hardcode
    def __init__(self, ns, scale=2):
        super(_MaskNetB_7_ns, self).__init__()

        self.scale = scale
        self.ns = ns

        self.head = nn.ModuleList()
        self.tail = nn.ModuleList()

        self.head.append(nn.Conv2d( 1, 32, 5, 1, 2)) #0
        self.head.append(nn.Conv2d(32, 16, 1, 1, 0)) #1
        
        self.core = CoreModule(self.ns)
        self.mask = MaskModule(self.ns)
        
        self.tail.append(nn.Conv2d(16, 32, 1, 1, 0)) #6
        self.tail.append(nn.Conv2d(32, scale * scale, 3, 1, 1)) #7:last layer
        
        # init_head:
        for i in range(len(self.head)):
            self.head[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.head[i].weight)

        # init_tail:
        for i in range(len(self.tail)):
            self.tail[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.tail[i].weight)

    def forward(self, x):
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        f = self.core.forward(z)

        z = F.relu(self.tail[0](f))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        return y

    def forward_merge_mask(self, x, masks: list, mask_out=False):
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        masks_out = []

        for ii in range(self.ns):
            p = self.mask.forward(z, ii)
            mask = torch.zeros_like(p)
            mask[p >  0.5] = 1.0

            z = self.core.forward(z, ii)

            if ii in masks:
                z = z * mask
                masks_out.append(mask)

        z = F.relu(self.tail[0](z))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        if mask_out:
            return y, masks_out

        return y

    def forward_stage_wise_sequential_train(self, x, masks: list, target_stage):
        # mask in masks are binary; 1.0 uses for C or branch 0, and vice versa
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        for ii in range(self.ns):
            p = self.mask.forward(z, ii)
            mask = torch.zeros_like(p)
            mask[p >  0.5] = 1.0

            z = self.core.forward(z, ii)

            if ii == target_stage:
                return z, p
            
            if ii in masks:
                z = z * mask
        
        z = F.relu(self.tail[0](z))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        return y
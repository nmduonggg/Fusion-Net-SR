"""
@author: nghiant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

class LargeModuleStage(nn.Module):
    def __init__(self):
        super(LargeModuleStage, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(64, 64, 3, 1, 1))

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x):
        z = x
        for i in range(len(self.conv)):
            z = F.relu(self.conv[i](z))
        return z

class LargeModule(nn.Module):
    def __init__(self, ns):
        super(LargeModule, self).__init__()

        self.ns = ns
        
        self.stages = nn.ModuleList()
        for i in range(ns):
            self.stages.append(LargeModuleStage())

    def forward(self, x, stages=[]):
        for s in stages:
            assert (s < self.ns) and (s >= 0), f"[ERRO] invalid stage {s}"

        z = x
        if stages:
            for s in range(self.ns):
                if s in stages:
                    z = self.stages[s](z)
            return z
        else:
            feas = []
            for s in range(self.ns):
                z = self.stages[s](z)
                feas.append(z)
            return z, feas

class SmallModuleStage(nn.Module):
    def __init__(self):
        super(SmallModuleStage, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(64, 16, 1, 1, 0))
        self.conv.append(nn.Conv2d(16, 64, 3, 1, 1))

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x):
        z = x
        for i in range(len(self.conv)):
            z = F.relu(self.conv[i](z))
        return z

class SmallModule(nn.Module):
    def __init__(self, ns):
        super(SmallModule, self).__init__()

        self.ns = ns

        self.stages = nn.ModuleList()
        for i in range(ns):
            self.stages.append(SmallModuleStage())

    def forward(self, x, stages=[]):
        for s in stages:
            assert (s < self.ns) and (s >= 0), f"[ERRO] invalid stage {s}"

        z = x
        if stages:
            for s in range(self.ns):
                if s in stages:
                    z = self.stages[s](z)
            return z
        else:
            feas = []
            for s in range(self.ns):
                z = self.stages[s](z)
                feas.append(z)
            return z, feas

class FusionNetB_8_4s(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(FusionNetB_8_4s, self).__init__()

        self.scale = scale
        self.ns = 4

        self.head = nn.ModuleList()
        self.branch = nn.ModuleList()
        self.tail = nn.ModuleList()

        self.head.append(nn.Conv2d( 1, 128, 5, 1, 2)) #0
        self.head.append(nn.Conv2d(128, 64, 1, 1, 0)) #1
        
        self.branch.append(LargeModule(self.ns))
        self.branch.append(SmallModule(self.ns))
        
        self.tail.append(nn.Conv2d(64, 128, 1, 1, 0)) #6
        self.tail.append(nn.Conv2d(128, scale * scale, 3, 1, 1)) #7:last layer
        
        # init_head:
        for i in range(len(self.head)):
            self.head[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.head[i].weight)

        # init_tail:
        for i in range(len(self.tail)):
            self.tail[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.tail[i].weight)

    def forward(self, x, branch=0, fea_out=False):
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        branch_fea, feas = self.branch[branch](z)

        z = F.relu(self.tail[0](branch_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        if fea_out:
            return y, feas

        return y

    def forward_merge_mask(self, x, masks: dict, fea_out=False):
        # mask in masks are binary; 1.0 uses for C or branch 0, and vice versa
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        feas = []

        for ii in range(self.ns):
            branch_fea_0 = self.branch[0](z, stages=[ii])
            branch_fea_1 = self.branch[1](z, stages=[ii])
            
            assert ii in masks, f"[ERRO]: missing mask for stage {ii}"

            merge_map = masks[ii]
            merge_fea = branch_fea_0 * merge_map + branch_fea_1 * (1.0 - merge_map)

            z = merge_fea
            feas += [branch_fea_0, branch_fea_1, merge_map]
        
        z = F.relu(self.tail[0](merge_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        if fea_out:
            return y, feas

        return y

    def forward_stage_wise_sequential_train(self, x, masks: dict, target_stage):
        # mask in masks are binary; 1.0 uses for C or branch 0, and vice versa
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        for ii in range(self.ns):
            branch_fea_0 = self.branch[0](z, stages=[ii])
            branch_fea_1 = self.branch[1](z, stages=[ii])

            if ii == target_stage:
                return branch_fea_0, branch_fea_1
            
            assert ii in masks, f"[ERRO]: missing mask for stage {ii}"

            merge_map = masks[ii]
            merge_fea = branch_fea_0 * merge_map + branch_fea_1 * (1.0 - merge_map)

            z = merge_fea
        
        z = F.relu(self.tail[0](merge_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        return y

    def forward_stage_wise_close_mask(self, x, mask_gen, p, target_stage):
        # mask_gen is a function
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        for ii in range(self.ns):
            
            merge_map = mask_gen.generate_mask(z, p)
            
            branch_fea_0 = self.branch[0](z, stages=[ii])
            branch_fea_1 = self.branch[1](z, stages=[ii])

            merge_fea = branch_fea_0 * merge_map + branch_fea_1 * (1.0 - merge_map)

            if ii == target_stage:
                return branch_fea_0, branch_fea_1, merge_map

            z = merge_fea
        
        z = F.relu(self.tail[0](merge_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        return y
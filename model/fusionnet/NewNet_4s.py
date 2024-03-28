"""
@author: nghiant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x
    
class ChannelAttentionBlock(nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()

        self.tau = 1.0
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, 1, 1, 0),
            nn.Sigmoid() 
        )

        self.conv_du[1].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.conv_du[1].weight)

        self.conv_du[3].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.conv_du[3].weight)

    def forward(self, x):
        ca_mask = self.conv_du(x)

        return ca_mask

class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()

        self.tau = 1.0
        self.spa_mask = nn.Sequential(
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.ReLU(True),
            # nn.AvgPool2d(2), #remove avg
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(True),
            # nn.ConvTranspose2d(8, 1, 3, 2, 1, output_padding=1), #dont use upsample with transpose
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.spa_mask[0].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.spa_mask[0].weight)

        self.spa_mask[2].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.spa_mask[2].weight)

        self.spa_mask[4].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.spa_mask[4].weight)

    def forward(self, x):
        sa_mask = self.spa_mask(x)

        return sa_mask

class NewSResBlock(nn.Module):
    def __init__(self):
        super(NewSResBlock, self).__init__()

        self.cab = ChannelAttentionBlock()
        self.sab = SpatialAttentionBlock()

        self.head = nn.Conv2d(32, 32, 3, 1, 1)

        self.tail = nn.Conv2d(32, 32, 3, 1, 1)

        self.head.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.head.weight)

        self.tail.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.tail.weight)

    def forward(self, x):
        ca_mask = self.cab(x)
        sa_mask = self.sab(x)

        z = F.relu(self.head(x))
        z = z * ca_mask

        z = self.tail(x)
        z = z * sa_mask

        z = F.relu(z + x)

        return z

class NewNet_4s(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(NewNet_4s, self).__init__()

        self.scale = scale
        self.ns = 4

        self.head = nn.ModuleList()
        self.body = nn.ModuleList()
        self.tail = nn.ModuleList()

        self.head.append(nn.Conv2d( 1, 64, 3, 1, 1))
        self.head.append(nn.Conv2d(64, 32, 1, 1, 0))
        
        for _ in range(self.ns):
            self.body.append(NewSResBlock())
        
        self.tail.append(nn.Conv2d(32, 64, 1, 1, 0))
        self.tail.append(nn.Conv2d(64, scale * scale, 3, 1, 1))
        
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

        for i in range(self.ns):
            z = self.body[i](z)

        z = F.relu(self.tail[0](z))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        return y
    
class ChannelAttentionBlock_2(nn.Module):
    '''Gumbel softmax version of Channel Attention Block'''
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()

        self.tau = 1.0
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32*2, 1, 1, 0),  # 2 layer for logits -> gumbel_softmax
        )

        self.conv_du[1].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.conv_du[1].weight)

        self.conv_du[3].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.conv_du[3].weight)

    def forward(self, x):
        ca_mask = self.conv_du(x)
        B, _, H, W = ca_mask.size()
        ca_mask.reshape(B, 2, -1, H, W)
        ca_mask = gumbel_softmax(ca_mask, dim=1, tau=self.tau)
        
        return ca_mask[:, 0, ...]   # reduce dim to BxCxHxW

class SpatialAttentionBlock_2(nn.Module):
    '''Gumbel softmax version of Spatial Attention Block'''
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()

        self.tau = 1.0
        self.spa_mask = nn.Sequential(
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.ReLU(True),
            # nn.AvgPool2d(2), #remove avg
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(8, 2, 3, 1, 1),   # 2 for softmax: logits -> gumbel softmax
        )

        self.spa_mask[0].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.spa_mask[0].weight)

        self.spa_mask[2].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.spa_mask[2].weight)

        self.spa_mask[4].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.spa_mask[4].weight)

    def forward(self, x):
        sa_mask = self.spa_mask(x)
        sa_mask = gumbel_softmax(sa_mask, dim=1, tau=self.tau)

        return sa_mask[:, :1, ...]  # choose 1 dim for dense, keep dims

class NewSResBlock(nn.Module):
    def __init__(self):
        super(NewSResBlock, self).__init__()

        self.cab = ChannelAttentionBlock()
        self.sab = SpatialAttentionBlock()

        self.head = nn.Conv2d(32, 32, 3, 1, 1)

        self.tail = nn.Conv2d(32, 32, 3, 1, 1)

        self.head.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.head.weight)

        self.tail.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.tail.weight)

    def forward(self, x):
        ca_mask = self.cab(x)
        sa_mask = self.sab(x)

        z = F.relu(self.head(x))
        z = z * ca_mask

        z = self.tail(x)
        z = z * sa_mask

        z = F.relu(z + x)

        return z

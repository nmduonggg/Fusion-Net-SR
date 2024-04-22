import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

import math

class SuperNet_share(nn.Module):
    def __init__(self, scale: int=2, tile:int=2):
        super(SuperNet_share, self).__init__()
        
        self.tile = tile
        self.scale = scale
        inp_channel = 1
        
        # heads
        self.heads = nn.Sequential(
            nn.Conv2d(inp_channel, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        # body
        self.body = BasicBlock(32, self.tile)
        # tail
        self.mask_predictors = nn.ModuleList([
            MaskPredictor(32) for _ in range(4)
        ])
        self.tails = nn.ModuleList([
            UpSampler(32, self.scale) for _ in range(4)
        ])
        
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(1, inp_channel, 1, 1)) for _ in range(4)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros((1, inp_channel, 1, 1))) for _ in range(4)
        ])
        
    
    def forward(self, x):
        x = self.heads(x)
        out_h, out_w = x.size(2)*self.scale, x.size(3)*self.scale
        densities = []
        masks = []
        outs = []
        
        for i in range(4):
            x, std_logit = self.body(x)
            outs.append(x*self.factors[i] + self.biases[i])
            std = self.mask_predictors[i](std_logit)
            masks.append(std)
        
        outs_mean = [self.tails[i](out) for i, out in enumerate(outs)]
        return [outs_mean, masks]
    
    def creat_MC_std(self, x, T):
        feat = self.heads(x)
        
        MC_outs = [[] for i in range(4)]
        with torch.no_grad():
            for t in range(T):
                x = feat    # restart
                for i in range(4):
                    x, _ = self.body(x)
                    x = x * self.factors[i] + self.biases[i]
                    MC_outs[i].append(self.tails[i](x).detach().cpu())
                
        MC_outs = [torch.stack(MC_out, dim=0) for MC_out in MC_outs]   # [TxBxCxHxW]x4 
        MC_std = [torch.std(MC_out, dim=0) for MC_out in MC_outs]   # -> [BxCxHxW]x4
        
        return MC_std
    
    def fuse_2_blocks(self, x, idxs, keep):
        """Fuse 2 blocks theoretically

        Args:
            idxs (list): List of indices of blocks
            keep (float): keep rate of 1st image

        Returns:
            out: fused image
        """
        assert 0 <= keep <= 1
        assert len(idxs)==2
        with torch.no_grad():
            yfs, masks = self.forward(x)
        # percentile filter - get the r% pixels with highest uncertainty
        hard_mask = masks[idxs[0]].clone().cpu().numpy()
        hard_mask = (hard_mask > np.percentile(hard_mask, keep*100)).astype(int)
        hard_mask = torch.tensor(hard_mask)
            
        hm = hard_mask.to(yfs[0].device)
        y = yfs[idxs[0]] * (1-hm) + yfs[idxs[1]] * hm
        
        return y
        
class BasicBlock(nn.Module):
    def __init__(self, channels, tile, scale=2):
        super(BasicBlock, self).__init__()
        self.channels = channels
        self.tile = tile
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False))

    def forward(self, x):
        B, C, H, W = x.size()
        shortcut = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut
        x = F.relu(x)
        variance_logit = F.interpolate(x, size=[H*self.scale, W*self.scale], mode='nearest')
        
        return [x, variance_logit]
    
class UpSampler(nn.Module):
    """Upsamler = Conv + PixelShuffle
    This class is hard-code for scale factor of 2"""
    def __init__(self, n_features, scale):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_features, scale*scale, 3, 1, 1))
        self.dropout = nn.Dropout(p=0.01)
        self.shuffler = nn.PixelShuffle(2)
        self.finalizer = nn.Conv2d(1, 1, 1, 1, 0)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.shuffler(x)
        x = self.finalizer(x)
        return x
    
class MaskPredictor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv(x) 
        return x

    

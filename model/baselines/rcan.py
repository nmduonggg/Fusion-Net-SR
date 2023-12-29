import torch 
import torch.nn as nn
from typing import Dict
from torch import Tensor


class RCAN(nn.Module):
    """
    Residual Channel Attention Net: [ResidualGroup]
    """
    def __init__(self, scale:int=2):
        super(RCAN, self).__init__()
        
        # head
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        
        # body
        res_groups = []
        for _ in range(4):
            res_groups.append(ResidualGroup(32, reduction=4, num_rcab=4))
        self.body = nn.Sequential(*res_groups)
        
        # after the extractor, reconnect a layer of conv blocks
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        
        # upsamling
        self.upsampling = UpSampler(32, scale)
        
        # output layer
        self.conv3 = nn.Conv2d(32, 1, 3, 1, 1)
        
        self.density = []
        
    def reset_density(self):
        self.density = []
        
    def forward(self, x):
        z = self.conv1(x)
        for i in range(4):
            x = self.body[i](z if i==0 else x)
            assert self.body[i].density is not None
            self.density.append(self.body[i].density)
        x = self.conv2(x)
        x = torch.add(x, z)
        x = self.upsampling(x)
        x = self.conv3(x)
        
        return x

class ResidualGroup(nn.Module):
    """
    Residual Group: [RCAB]
    """
    def __init__(self, channel: int, reduction: int, num_rcab: int):
        super(ResidualGroup, self).__init__()
        residual_group = []
        for _ in range(num_rcab):
            residual_group.append(RCAB(channel, reduction))
        residual_group.append(nn.Conv2d(channel, channel, 3, 1, 1))
        self.num_rcab = num_rcab
        self.residual_group = nn.Sequential(*residual_group)
        self.density = None
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.residual_group(x)
        out = torch.add(out, identity)
        
        for i, block in enumerate(self.residual_group[:-1]):
            self.density = block.density if i==0 \
                    else block.density + self.density
        self.density /= self.num_rcab
        return out
        

class RCAB(nn.Module):
    """
    Residual Channel Attention Block
    """
    def __init__(self, channel:int, reduction:int):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            ChannelAttention(channel, reduction)
        )
        self.density = None
        
    def forward(self, x):
        identity = x
        out = self.body(x)
        self.density = self.body[-1].mask.mean()
        out = out + identity
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channel: int, reduction: int=4):
        super(ChannelAttention, self).__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, 1, 1, 0),
            nn.Sigmoid(),   # actually generate both spatial and channel mask, but each channel has its own spatial mask
        )
        self.mask = None

    def forward(self, x: Tensor) -> Tensor:
        out = self.body(x)
        self.mask = out.contiguous().cpu()
        self.mask = (self.mask >= 0.5).float()
        out = torch.mul(out, x)

        return out

class UpSampler(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpSampler, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, 3, 1, 1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x

if __name__=='__main__':
    from utils import calc_flops
    model = RCAN(2)
    calc_flops(model)
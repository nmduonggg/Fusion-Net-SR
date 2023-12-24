import torch 
import torch.nn as nn
from typing import Dict
from torch import Tensor

class EDSR(nn.Module):
    def __init__(self, scale:int=2):
        super().__init__()
        
        # common blocks
        self.heads = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), 
            nn.Conv2d(64, 32, 1, 1, 0)
        )
        self.body = nn.Sequential(*[
            ResBlock(32, 3, 1, 1) for _ in range(4) 
        ])
        self.tails = nn.Sequential(UpSampler(32), nn.Conv2d(32, 1, 1, 1, 0))
        
    def forward(self, x: Tensor) -> Tensor:
        z = self.heads(x)
        z = self.body(z)
        z = self.tails(z)
        
        return z
    
class ResBlock(nn.Module):
    """Baseline of EDSR so rescale factor is removed (original paper set up is 0.1 for deep architecture)
    """
    def __init__(self, n_features, kernel_size=3, stride=1, padding=1, bias=False, rescale_factor = 1.):
        super().__init__()
        self.rescale_factor = rescale_factor
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        z = x
        residual = self.relu(self.conv1(x))
        residual = self.conv2(x).mul(self.rescale_factor)
        y = z + residual
        
        return y
    
class UpSampler(nn.Module):
    """Upsamler = Conv + PixelShuffle
    This class is hard-code for scale factor of 2"""
    def __init__(self, n_features, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 4*n_features, 3, 1, 1, bias=bias)
        self.shuffler = nn.PixelShuffle(2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffler(x)
        return x
    
if __name__=='__main__':
    from utils import calc_flops
    model = EDSR(2)
    calc_flops(model)
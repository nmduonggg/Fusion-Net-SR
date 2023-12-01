import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math

class DGNet(nn.Module):
    def __init__(self, scale: int=2):
        super(DGNet, self).__init__()
        
        # lr patch size
        self.h = 21
        self.w = 21
        self.block_h = 3
        self.block_w = 3
        # heads
        self.heads = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.Conv2d(64, 32, 1, 1, 0)
        )
        # body
        self.body = nn.Sequential(*[
            BasicBlock(32, self.h, self.w, self.block_h, self.block_w) for _ in range(25)\
        ])
        # tail
        self.tails = nn.Sequential(UpSampler(32), nn.Conv2d(32, 1, 1, 1, 0))
    
    def forward(self, x):
        x = self.heads(x)
        x = self.body(x)
        x = self.tails(x)
        return x
        
class BasicBlock(nn.Module):
    def __init__(self, channels, h, w, block_h, block_w):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels))
        self.mask_c = ChannelAttention(channels, channels)
        self.mask_s = SpatialAttention(h, w, channels, block_h, block_w)
        self.upsampler = nn.Upsample(size=[h, w], mode='nearest')
    
    def forward(self, x):
        residual = x
        masked_c, _, _ = self.mask_c(x)
        masked_s = self.upsampler(self.mask_s(x)[0])
        x = self.conv1(x)
        x = x * masked_c * masked_s if not self.training else x * masked_c
        x = self.conv2(x)
        x = F.relu(x * masked_s + residual)
        return x
    
class UpSampler(nn.Module):
    """Upsamler = Conv + PixelShuffle
    This class is hard-code for scale factor of 2"""
    def __init__(self, n_features, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 4*n_features, 3, 1, 1, bias=bias)
        self.shuffler = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffler(x)
        x = self.relu(x)
        return x
        
class GumbelSoftmax(nn.Module):
    '''
        Gumbel softmax gate
    '''
    def __init__(self, tau=1):
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.sigmoid = nn.Sigmoid()
        
    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone.uniform_()
        gumbel_samples_tensor = torch.log(uniform_samples_tensor+eps) - torch.log(1-uniform_samples_tensor+eps)
        return gumbel_samples_tensor
    
    def gumbel_softmax(self, logits):
        """draw a sample from gumbel-softmax distribution
        """
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.tau)
        
        return soft_samples, logits
    
    def forward(self, logits):
        if not self.training:
            out_hard = (logits>=0).float()
            return out_hard
        
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard


class SpatialAttention(nn.Module):
    '''
        Spatial Attention.
    '''
    def __init__(self, h, w, planes, block_w, block_h, eps=0.66667,
                 bias=-1, **kwargs):
        super(SpatialAttention, self).__init__()
        # Parameter
        self.width, self.height, self.channel = w, h, planes
        self.mask_h, self.mask_w = int(np.ceil(h / block_h)), int(np.ceil(w / block_w))
        self.eleNum_s = torch.Tensor([self.mask_h*self.mask_w])
        # spatial attention
        self.atten_s = nn.Conv2d(planes, 1, kernel_size=3, stride=1, bias=bias>=0, padding=1)
        if bias>=0:
            nn.init.constant_(self.atten_s.bias, bias)
        # Gate
        self.gate_s = GumbelSoftmax(tau=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1,2,3))
    
    def forward(self, x):
        batch, channel, height, width = x.size()
        # Pooling
        input_ds = F.adaptive_avg_pool2d(input=x, output_size=(self.mask_h, self.mask_w))
        # spatial attention
        s_in = self.atten_s(input_ds) # [N, 1, h, w]
        # spatial gate
        mask_s = self.gate_s(s_in) # [N, 1, h, w]
        # norm
        norm = self.norm(mask_s)
        norm_t = self.eleNum_s.to(x.device)
        return mask_s, norm, norm_t
    
    def get_flops(self):
        flops = self.mask_h * self.mask_w * self.channel * 9
        return flops


class ChannelAttention(nn.Module):
    '''
        Attention Mask.
    '''
    def __init__(self, inplanes, outplanes, fc_reduction=4, eps=0.66667, bias=-1, **kwargs):
        super(ChannelAttention, self).__init__()
        # Parameter
        self.bottleneck = inplanes // fc_reduction 
        self.inplanes, self.outplanes = inplanes, outplanes
        self.eleNum_c = torch.Tensor([outplanes])
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.atten_c = nn.Sequential(
            nn.Conv2d(inplanes, self.bottleneck, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck, outplanes, kernel_size=1, stride=1, bias=bias>=0),
        )
        if bias>=0:
            nn.init.constant_(self.atten_c[3].bias, bias)
        # Gate
        self.gate_c = GumbelSoftmax(tau=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1,2,3))
    
    def forward(self, x):
        batch, channel, _, _ = x.size()
        context = self.avg_pool(x) # [N, C, 1, 1] 
        # transform
        c_in = self.atten_c(context) # [N, C_out, 1, 1]
        # channel gate
        mask_c = self.gate_c(c_in) # [N, C_out, 1, 1]
        # norm
        norm = self.norm(mask_c)
        norm_t = self.eleNum_c.to(x.device)
        return mask_c, norm, norm_t
    
    def get_flops(self):
        flops = self.inplanes * self.bottleneck + self.bottleneck * self.outplanes
        return flops
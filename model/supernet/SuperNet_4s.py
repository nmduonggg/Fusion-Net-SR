import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

import math

class SuperNet(nn.Module):
    def __init__(self, scale: int=2, tile:int=2):
        super(SuperNet, self).__init__()

        self.tile = tile
        self.scale = scale
        
        # heads
        self.heads = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        # body
        self.body = nn.ModuleList([
            BasicBlock(32, self.tile, self.scale) for _ in range(4)])
        # tail
        self.tails = nn.ModuleList([
            UpSampler(32, self.scale) for _ in range(4)
        ])
        
        self.density = []
        self.masked_s = []
        
    def reset_density(self):
        self.density = []
        
    def reset_mask(self):
        self.masked_s = []
    
    def forward(self, x):
        x = self.heads(x)
        out_h, out_w = x.size(2)*self.scale, x.size(3)*self.scale
        densities = []
        masks = []
        outs = []
        
        for i in range(4):
            x, mask = self.body[i](x)
            outs.append(x)
            masks.append(mask)
            
        # masks = torch.cat(masks, dim=1) # skip mask of last block
        
        outs = [self.tails[i](out) for i, out in enumerate(outs)]
        # print(masks.shape)
        return [outs, masks]
    
    def forward_by_stage(self, x, nblocks):
        """
        Forward by block stage
        ::in
            x: input tensor
            stage_id: end id of block
        ::out
            x: output tensor
            density: density value in range [0, 1]
            mask_s: spatial mask prediction
        """
        out_h, out_w = x.size(2)*self.scale, x.size(3)*self.scale
        if nblocks==-1: nblocks = len(self.body)
        assert nblocks <= len(self.body) and nblocks > 0, f"Max number of block is {len(self.body)}, got {nblocks}"
        x = self.heads(x)
        for idx in range(nblocks):
            x, density, mask_s = self.body[idx](x)
        x = self.tails(x)
        mask_s = F.interpolate(mask_s, size=[out_h, out_w], mode="nearest")
        return x, density, mask_s
        
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
            nn.Conv2d(channels, channels, 3, 1, 1))
        self.mask_predictor = MaskPredictor(channels)
    
    def forward(self, x):
        B, C, H, W = x.size()
        shortcut = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        mask = self.mask_predictor(F.interpolate(x, size=[H*self.scale, W*self.scale], mode='nearest'))
        
        x = x + shortcut
        x = F.relu(x)
        
        return [x, mask]
    
class UpSampler(nn.Module):
    """Upsamler = Conv + PixelShuffle
    This class is hard-code for scale factor of 2"""
    def __init__(self, n_features, scale):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_features, scale*scale, 3, 1, 1))
        self.shuffler = nn.PixelShuffle(2)
        self.finalizer = nn.Conv2d(1, 1, 1, 1, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffler(x)
        x = self.finalizer(x)
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
        uniform_samples_tensor = template_tensor.clone().uniform_()
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
    
class MaskPredictor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, 1, 1), nn.ELU(),
            nn.Conv2d(channels//4, 1, 3, 1, 1), nn.ELU())
    def forward(self, x):
        x = self.conv(x)
        return x
    
class SpatialAttention(nn.Module):
    '''
        Spatial Attention.
    '''
    def __init__(self, channels, tile, eps=0.66667,
                 bias=-1, **kwargs):
        super(SpatialAttention, self).__init__()
        self.channel = channels
        self.tile = tile
        # spatial attention
        self.atten_s = nn.Conv2d(channels, 1, kernel_size=3, stride=1, bias=bias>=0, padding=1)
        if bias>=0:
            nn.init.constant_(self.atten_s.bias, bias)
        # Gate
        self.gate_s = GumbelSoftmax(tau=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1,2,3))
        self.avgpool = nn.AvgPool2d(kernel_size=tile, stride=tile)
    
    def forward(self, x):
        # Pooling
        input_ds = self.avgpool(x)
        # spatial attention
        s_in = self.atten_s(input_ds) # [N, 1, h, w]
        # spatial gate
        mask_s = self.gate_s(s_in) # [N, 1, h, w]
        return mask_s, s_in
    
class ChannelAttention(nn.Module):
    '''
        Attention Mask.
    '''
    def __init__(self, inplanes, outplanes, fc_reduction=4, eps=0.66667, bias=-1, **kwargs):
        super(ChannelAttention, self).__init__()
        # Parameter
        self.bottleneck = inplanes // fc_reduction 
        self.inplanes, self.outplanes = inplanes, outplanes

        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.atten_c = nn.Sequential(
            nn.Conv2d(inplanes, self.bottleneck, kernel_size=1, stride=1, bias=False),
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
        context = self.avg_pool(x) # [N, C, 1, 1] 
        # transform
        c_in = self.atten_c(context) # [N, C_out, 1, 1]
        # channel gate
        mask_c = self.gate_c(c_in) # [N, C_out, 1, 1]
        
        return mask_c, c_in
    
    
if __name__=='__main__':
    from utils import calc_flops
    model = DGNetSR(2, 1)
    # model.load_state_dict(torch.load('/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/DGNet/_best.t7', map_location='cpu'))
    calc_flops(model)
    
    
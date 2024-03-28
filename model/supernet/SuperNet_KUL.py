import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

import math

class SuperNet_kul(nn.Module):
    def __init__(self, scale: int=2, tile:int=2):
        super(SuperNet_kul, self).__init__()

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
            x, std = self.body[i](x)
            outs.append(x)
            masks.append(std)
            
        # masks = torch.cat(masks, dim=1) # skip mask of last block
        
        outs_mean = [self.tails[i](out) for i, out in enumerate(outs)]
        # print(masks.shape)
        return [outs_mean, masks]
    
    def creat_MC_std(self, x, T):
        feat = self.heads(x)
        
        MC_outs = [[] for i in range(4)]
        with torch.no_grad():
            for t in range(T):
                x = feat    # restart
                for i in range(4):
                    x, _ = self.body[i](x)
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
            nn.Conv2d(channels, channels, 3, 1, 1))
        self.mask_predictor = MaskPredictor(channels)

    def forward(self, x):
        B, C, H, W = x.size()
        shortcut = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut
        x = F.relu(x)
        variance = self.mask_predictor(F.interpolate(x, size=[H*self.scale, W*self.scale], mode='nearest'))
        
        return [x, variance]
    
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
            nn.Conv2d(channels, 1, 3, 1, 1))
        
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
    
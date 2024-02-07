import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

import math

class DGNetSR(nn.Module):
    def __init__(self, scale: int=2, tile:int=2):
        super(DGNetSR, self).__init__()

        self.tile = tile
        
        # heads
        self.heads = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.Conv2d(64, 32, 1, 1, 0)
        )
        # body
        self.body = nn.ModuleList([
            BasicBlock(32, self.tile) for _ in range(4)])
        # tail
        self.tails = nn.Sequential(
            UpSampler(32), nn.Conv2d(32, 1, 1, 1, 0))
        self.density = []
        
    def reset_density(self):
        self.density = []
        
    # def flop_forward(self, x, label, den_target, lbda, gamma, p):
    #     batch_num, C, H, W = x.shape
    #     flops_heads = torch.Tensor([1*64*9 *H*W + 64*32*9*H*W])
    #     x = self.heads(x)
    #     # residual modules
    #     norm1 = torch.zeros(1, batch_num+1).to(x.device)
    #     norm2 = torch.zeros(1, batch_num+1).to(x.device)
    #     flops = torch.zeros(1, batch_num+2).to(x.device)    # flops[0] = 0
    #     x = (x, norm1, norm2, flops)
    #     for i in range(3):
    #         x = self.body[i].flop_forward(x)
    #         self.density.append(self.body[i].density.cpu())
            
    #     x, norm1, norm2, flops = self.body[3].flop_forward(x)
    #     self.density.append(self.body[3].density)
    #     x = self.tails(x)
        
    #     # norm and flops
    #     norm_s = norm1[1:, 0:batch_num].permute(1, 0).contiguous()
    #     norm_c = norm2[1:, 0:batch_num].permute(1, 0).contiguous()
    #     norm_s_t = norm1[1:, -1].unsqueeze(0)
    #     norm_c_t = norm2[1:, -1].unsqueeze(0)
        
        
    #     # flops_real = [[flop_masked_conv (real), flop_mask, flop_conv_full (original)], [2], ...]
    #     flops_real = [flops[1:, 0:batch_num].permute(1, 0).contiguous(), 
    #                   flops_heads.to(x.device)]
    #     flops_mask = flops[1:, -2].unsqueeze(0)
    #     flops_ori  = flops[1:, -1].unsqueeze(0)
    #     # get outputs
    #     outputs = {}
    #     outputs["closs"], outputs["rloss"], outputs["bloss"] = self.get_loss(
    #                         x, label, batch_num, den_target, lbda, gamma, p,
    #                         norm_s, norm_c, norm_s_t, norm_c_t, 
    #                         flops_real, flops_mask, flops_ori)
    #     outputs["out"] = x
    #     outputs["flops_real"] = flops_real
    #     outputs["flops_mask"] = flops_mask
    #     outputs["flops_ori"] = flops_ori
    #     return outputs
    
    # def set_criterion(self, criterion):
    #     self.criterion = criterion
    #     return
    
    # def get_loss(self, output, label, batch_size, den_target, lbda, gamma, p,
    #              mask_norm_s, mask_norm_c, norm_s_t, norm_c_t,
    #              flops_real, flops_mask, flops_ori):
    #     closs, rloss, bloss = self.criterion(output, label, flops_real, flops_mask,
    #             flops_ori, batch_size, den_target, lbda, mask_norm_s, mask_norm_c,
    #             norm_s_t, norm_c_t, gamma, p)
    #     return closs, rloss, bloss
        
    
    def forward(self, x):
        x = self.heads(x)
        densities = []
        for i in range(4):
            x, density = self.body[i](x)
            densities.append(density)
            dense = self.body[i].density
            assert dense is not None
            self.density.append(dense)
        densities = torch.stack(densities, dim=0).mean()
        x = self.tails(x)
        return [x, densities]
        
class BasicBlock(nn.Module):
    def __init__(self, channels, tile):
        super(BasicBlock, self).__init__()
        self.channels = channels
        self.tile = tile
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels))
        self.mask_c = ChannelAttention(channels, channels)
        self.mask_s = SpatialAttention(channels, tile)
        self.density = None
        
        # # mask flops
        # flops_mks = self.mask_s.get_flops()
        # flops_mkc = self.mask_c.get_flops()
        # self.flop_mask = torch.Tensor([flops_mks + flops_mkc])
        
    # def flop_forward(self, input):
    #     x, norm_1, norm_2, flops = input
    #     B, C, H, W = x.shape
        
    #     # flops
    #     flops_conv1_full = torch.Tensor([9 * H * W * self.channels * self.channels])
    #     flops_conv2_full = torch.Tensor([9 * H + W * self.channels * self.channels])
    #     flops_full = flops_conv1_full + flops_conv2_full
        
    #     # forwarding
    #     residual = x
    #     mask_s_m, norm_s, norm_s_t = self.mask_s(x)
    #     mask_c, norm_c, norm_c_t = self.mask_c(x)
    #     mask_s = F.interpolate(mask_s_m, size=[H, W], mode='nearest')
    #     out = self.conv1(x)
    #     out = out * mask_c * mask_s if not self.training else out * mask_c
    #     # conv 2
    #     out = self.conv2(out)
    #     out = out * mask_s
    #     out = out + residual
    #     out = F.relu(out)
        
    #     # norm 
    #     norm_1 = torch.cat((norm_1, torch.cat((norm_s, norm_s_t)).unsqueeze(0)))
    #     norm_2 = torch.cat((norm_2, torch.cat((norm_c, norm_c_t)).unsqueeze(0)))
        
    #     # flops
    #     flops_blk = self.get_flops(mask_s, mask_c, flops_full)
    #     flops = torch.cat((flops, flops_blk.unsqueeze(0)))  # stack by Lth layer
        
    #     s = x.contiguous().cpu()
    #     self.density = (s > 0).float().mean()        
        
    #     return (out, norm_1, norm_2, flops)
    
    def forward(self, x):
        B, C, H, W = x.size()
        residual = x
        masked_c = self.mask_c(x)
        masked_s = self.mask_s(x)
        masked_s = F.interpolate(masked_s, size=[H, W], mode='nearest')
        x = self.conv1(x)
        x = x * masked_c * masked_s if not self.training else x * masked_c
        x = self.conv2(x)
        x = F.relu(x * masked_s + residual)
        s = x.contiguous().cpu()
        self.density = (s > 0).float().mean()
        dense_mask = masked_c * masked_s
        return [x, dense_mask]
    
    # def get_flops(self, mask_s_up, mask_c, flops_full):
    #     # [B, C, H, W] -> [B]
    #     s_sum = mask_s_up.sum((1,2,3))  
    #     c_sum = mask_c.sum((1,2,3)) # [B]: number of activated points ?
    #     # conv1
    #     flops_conv1 = 9 * s_sum * c_sum * self.channels
    #     # conv2
    #     flops_conv2 = 9 * s_sum * c_sum * self.channels
    #     # total
    #     flops = flops_conv1 + flops_conv2   # theoretical flops after masked ?
    #     return torch.cat((flops, self.flop_mask.to(flops.device), flops_full.to(flops.device)))    # flops_conv_masked, flops_mask, flops_conv_full (ori)
    
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
class SpatialAttention(nn.Module):
    '''
        Spatial Attention.
    '''
    # def __init__(self, h, w, channels, block_h, block_w, eps=0.66667,
    #              bias=-1, **kwargs):
    #     super(SpatialAttention, self).__init__()
    #     self.height, self.width, self.channel = h, w, channels
    #     self.mask_h, self.mask_w = int(np.ceil(h / block_h)), int(np.ceil(w / block_w))
    #     self.eleNum_s = torch.Tensor([self.mask_h*self.mask_w])
    #     # spatial attention
    #     self.atten_s = nn.Conv2d(channels, 1, kernel_size=3, stride=1, bias=bias>=0, padding=1)
    #     if bias>=0:
    #         nn.init.constant_(self.atten_s.bias, bias)
    #     # Gate
    #     self.gate_s = GumbelSoftmax(tau=eps)
    #     # Norm
    #     self.norm = lambda x: torch.norm(x, p=1, dim=(1,2,3))
    #     self.avgpool = nn.AdaptiveAvgPool2d(output_size=(self.mask_h, self.mask_w))
    
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
        # self.mask_h, self.mask_w = input_ds.shape[-2:]
        # eleNum_s = torch.Tensor([self.mask_h*self.mask_w])
        # spatial attention
        s_in = self.atten_s(input_ds) # [N, 1, h, w]
        # spatial gate
        mask_s = self.gate_s(s_in) # [N, 1, h, w]
        # # norm
        # norm = self.norm(mask_s)
        # norm_t = eleNum_s.to(x.device)
        return mask_s
    
    # def get_flops(self):
    #     # assert self.mask_h > 0 and self.mask_w > 0
    #     flops = self.mask_h * self.mask_w * self.channel * 9
    #     return flops

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
        context = self.avg_pool(x) # [N, C, 1, 1] 
        # transform
        c_in = self.atten_c(context) # [N, C_out, 1, 1]
        # channel gate
        mask_c = self.gate_c(c_in) # [N, C_out, 1, 1]
        # norm
        # norm = self.norm(mask_c)
        # norm_t = self.eleNum_c.to(x.device)
        return mask_c
    
    # def get_flops(self):
    #     flops = self.inplanes * self.bottleneck + self.bottleneck
    #     return flops
    
###### LOSS ######
class blance_loss(nn.Module):
    def __init__(self):
        super(blance_loss, self).__init__()
        
    def forward(self, mask_norm_s, mask_norm_c, norm_s_t, norm_c_t, batch_size, den_target, gamma, p):
        norm_s = mask_norm_s
        norm_s_t = norm_s_t.mean(0)
        norm_c = mask_norm_c
        norm_c_t = norm_c_t.mean(0)
        den_s = norm_s[0:batch_size,:].mean(0) / norm_s_t
        den_c = norm_c[0:batch_size,:].mean(0) / norm_c_t
        den_tar = math.sqrt(den_target)
        bloss_s = self.get_bloss_basic(den_s, den_tar, batch_size, gamma, p)
        bloss_c = self.get_bloss_basic(den_c, den_tar, batch_size, gamma, p)
        bloss = bloss_s + bloss_c
        return bloss

    def get_bloss_basic(self, spar, spar_target, batch_size, gamma, p):
        # bounding loss
        bloss_l = (F.relu(p*spar_target - spar)**2).mean()
        bloss_u = (F.relu(spar-1 + p - p*spar_target)**2).mean()
        bloss = gamma * (bloss_l + bloss_u)
        return bloss
    
class spar_loss(nn.Module):
    def __init__(self):
        super(spar_loss, self).__init__()
    
    def forward(self, flops_real, flops_mask, flops_ori, batch_size, den_target, lbda):
        # total sparsity
        flops_tensor, flops_heads = flops_real[0], flops_real[1]
        # block flops
        flops_conv = flops_tensor[0:batch_size,:].mean(0).sum()
        flops_mask = flops_mask.mean(0).sum()
        flops_ori = flops_ori.mean(0).sum() + flops_heads.mean()
        flops_real = flops_conv + flops_mask + flops_heads.mean() 
        # loss
        rloss = lbda * (flops_real / flops_ori - den_target)**2
        return rloss
            
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.task_loss = nn.L1Loss()
        self.spar_loss = spar_loss()
        self.balance_loss = blance_loss()
        
    def forward(self, output, targets, flops_real, flops_mask, \
                flops_ori, batch_size, den_target, lbda, mask_norm_s, \
                mask_norm_c, norm_s_t, norm_c_t, gamma, p):
        closs = self.task_loss(output, targets)
        sloss = self.spar_loss(flops_real, flops_mask, flops_ori,\
                                batch_size, den_target, lbda)
        bloss = self.balance_loss(mask_norm_s, mask_norm_c, norm_s_t,\
                                norm_c_t, batch_size, den_target, gamma, p)
        return closs, sloss, bloss
    
if __name__=='__main__':
    from utils import calc_flops
    model = DGNet(2)
    model.load_state_dict(torch.load('/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/DGNet/_best.t7', map_location='cpu'))
    calc_flops(model)
    
    
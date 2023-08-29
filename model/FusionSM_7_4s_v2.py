import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    # make random noise fullfilled in noise matrix
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log() # gumbel distribution
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        # global average pooling: feature -> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale -> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y
        
class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, ns=4):
        super(MaskedConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.tau = 1
        self.relu = nn.ReLU(True)

        # channel mask
        self.ch_mask = nn.Parameter(torch.rand(1, out_channels, 2))

        # body conv
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.conv.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.conv.weight)
        
        # collect information after concat
        # self.collect = nn.Conv2d(out_channels * ns, out_channels, 1, 1, 0) #6

    def _update_tau(self, tau):
        self.tau = tau

    def _prepare(self): # apply channel mask to sparse conv
        # channel mask
        ch_mask = self.ch_mask.softmax(2).round()
        self.ch_mask_round = ch_mask

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []
        # 1 dense 0 sparse
           
        self.d_in_num.append(self.in_channels)
        self.s_in_num.append(0)
        self.d_out_num.append(int(ch_mask[0, :, 1].sum(0)))
        self.s_out_num.append(int(ch_mask[0, :, 0].sum(0)))

        # kernel split
        kernel_d2d = []
        kernel_d2s = []
        kernel_s = []

        # filter weights corresponding to channel mask
        #NOTE: At stage s:
        ##       - channels of each filter based on ch_mask[s-1]
        ##       - number of filters based on ch_mask[s]

        kernel_s.append([])
        if self.d_out_num[0] > 0:
            kernel_d2d.append(self.conv.weight[ch_mask[0, :, 1]==1, ...].view(self.d_out_num[0], -1))
        else:
            kernel_d2d.append([])
        if self.s_out_num[0] > 0:
            kernel_d2s.append(self.conv.weight[ch_mask[0, :, 0]==1, ...].view(self.s_out_num[0], -1))

        self.kernel_d2d = kernel_d2d
        self.kernel_d2s = kernel_d2s
        self.kernel_s = kernel_s


    def _generate_indices(self): # apply spatial mask to sparse conv
        A = torch.arange(3).to(self.spa_mask.device).view(-1, 1, 1)
        mask_indices = torch.nonzero(self.spa_mask.squeeze()) # in this function, 1 is sparse

        # indices: dense to sparse (1x1)
        self.h_idx_1x1 = mask_indices[:, 0] #x index of sparse positions
        self.w_idx_1x1 = mask_indices[:, 1] #y

        # indices: dense to sparse (3x3)
        mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A #NOTE: Why + A

        self.h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1) # 1 1 1 2 2 2 3 3 3 
        self.w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1) # 1 2 3 1 2 3 1 2 3

        # indices: sparse to sparse (3x3)
        indices = torch.arange(float(mask_indices.size(0))).view(1, -1).to(self.spa_mask.device) + 1
        self.spa_mask[0, 0, self.h_idx_1x1, self.w_idx_1x1] = indices # replace sparse position to idx from 1 to S

        self.idx_s2s = F.pad(self.spa_mask, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9, -1).long()

    def _mask_select(self, x, k):
        if k == 1:
            return x[0, :, self.h_idx_1x1, self.w_idx_1x1] # take every positions in all channels
        if k == 3:
            return F.pad(x, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9 * x.size(1), -1)
        
    def _sparse_conv(self, fea_dense, k, index=0):
        '''
        Perform sparse convolution at a specific stage, the result might be None in the case that there are lack of 
        sparse/dense channel at that stage

        Args:
        :param fea_dense: (B, C_d, H, W)
        :param fea_sparse: (C_s, N)
        :param k: kernel size
        :param index: stage index

        Returns: feature_dense and feature_sparse after the conv at stage
        '''
        assert (index==0), "Only support for 1 layer (index=0)"
        ### dense input ###
        
        # transform to patches for conv in linear order
        fea_col = F.unfold(fea_dense, k, stride=1, padding=(k-1) // 2).squeeze(0) 
        # matmul corresponding weight
        fea_d2d = torch.mm(self.kernel_d2d[index].view(self.d_out_num[index], -1), fea_col) 
        fea_d2d = fea_d2d.view(1, self.d_out_num[index], fea_dense.size(2), fea_dense.size(3)) # 1, dout, H', W'

        fea_d2s = torch.mm(self.kernel_d2s[index].view(self.s_out_num[index], -1), fea_col)
        fea_d2s = fea_d2s.view(1, self.s_out_num[index], fea_dense.size(2), -1)

        # dense to sparse
        fea_d2s_masked = torch.mm(self.kernel_d2s[index], self._mask_select(fea_dense, k))
            
        ### fusion v2 ###
        fea_d2s[0, :, self.h_idx_3x3, self.w_idx_3x3] = fea_d2s_masked
        fea_d = torch.cat([fea_d2d, fea_d2s], 1)
        print(fea_d.size())
        
        return fea_d
    
    def forward(self, x):
        '''
        :param x: [x[0], x[1]]
        x[0]: input feature [B, C, H, W]
        x[1]: spatial mask [B, 1, H, W]
        '''

        self._prepare()

        if self.training:
            spa_mask = x[1]
            ch_mask = gumbel_softmax(self.ch_mask, 3, self.tau)

            fea = x[0]
            fea = self.conv[0](fea)
            fea = fea * ch_mask[:, :, :1] * spa_mask + fea * ch_mask[:, :, 1:]
                    
            fea = self.relu(fea)

            return fea, ch_mask
        
        if not self.training:
            self.spa_mask = x[1]

            # generate indices
            self._generate_indices()

            # sparse conv
            fea_d = x[0]
            fea_s = None

            fea_d = self._sparse_conv(fea_d, fea_s, k = 3)
            out = self.relu(fea_d)
          
            return out, self.ch_mask
        
class LargeModule(nn.Module):
    def __init__ (self, ns):
        super(LargeModule, self).__init__()
        self.ns = ns
        self.tau = 1

        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(16, 4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 2, 3, 2, 1, output_padding=1),
        )

        self.body = nn.ModuleList()
        for _ in range(self.ns):
            self.body.append(MaskedConv2d(16, 16))
        
    def _update_tau(self, tau):
        self.tau = tau

    def forward(self, x, stages=[]):
        # TODO: Write forward
        for s in stages:
            assert (s < self.ns) and (s >= 0), f"[ERROR] invalid stage {s}"
        z = x
        ch_masks = []
        if stages:
           
            if self.training:
                spa_mask = self.spa_mask(z)
                spa_mask = gumbel_softmax(spa_mask, 1, self.tau)                
                for s in stages:
                    z, ch_mask = self.body([z, spa_mask[:, :1, ...]])
                    ch_masks.append(ch_mask.unsqueeze(2))
                ch_masks = torch.cat(ch_masks, 2)
                return z, ch_masks
            
            if not self.training:
                spa_mask = self.spa_mask(x)
                spa_mask = (spa_mask[:, :1, ...] > spa_mask[:, 1:, ...]).float()

                for s in stages:
                    z, ch_mask = self.body([z, spa_mask])

                return z, ch_mask # ch_mask are not used in inference
        
        else:
           
            if self.training:
                spa_mask = self.spa_mask(z)
                spa_mask = gumbel_softmax(spa_mask, 1, self.tau)                
                for s in range(self.ns):
                    z, ch_mask = self.body[s]([z, spa_mask[:, :1, ...]])
                    ch_masks.append(ch_mask.unsqueeze(2))
                ch_masks = torch.cat(ch_masks, 2)
                return z, ch_masks
            
            if not self.training:
                spa_mask = self.spa_mask(x)
                spa_mask = (spa_mask[:, :1, ...] > spa_mask[:, 1:, ...]).float()

                for s in range(self.ns):
                    z, ch_mask = self.body[s]([z, spa_mask])

                return z, ch_mask # ch_mask are not used in inference
    
class SmallModule(nn.Module):
    def __init__(self, ns):
        super(SmallModule, self).__init__()

        self.ns = ns

        self.conv = nn.ModuleList()
        for i in range(ns):
            self.conv.append(nn.Conv2d(16, 4, 1, 1, 0))
            self.conv.append(nn.Conv2d(4, 16, 3, 1, 1))

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x, stages=[]):
        for s in stages:
            assert (s < self.ns) and (s >= 0), f"[ERRO] invalid stage {s}"

        z = x
        if stages:
            for s in range(self.ns):
                if s in stages:
                    z = F.relu(self.conv[2*s](z))
                    z = F.relu(self.conv[2*s+1](z))
            return z
        else:
            feas = []
            for s in range(self.ns):
                z = F.relu(self.conv[2*s](z))
                z = F.relu(self.conv[2*s+1](z))
                feas.append(z)
            return z, feas
        

class FusionSM_7_4s_v2(nn.Module): #hardcode
    def __init__(self, ns=4, scale=2):
        super(FusionSM_7_4s_v2, self).__init__()

        self.scale = scale
        self.ns = 4 # for testing

        self.head = nn.ModuleList() # feature map 
        self.branch = nn.ModuleList() # 2 branches: Simple and Complex
        self.tail = nn.ModuleList() # concate output features
        self.mask = nn.ModuleList() # mask before put into branches

        self.head.append(nn.Conv2d( 1, 32, 5, 1, 2)) #0
        self.head.append(nn.Conv2d(32, 16, 1, 1, 0)) #1

        self.branch.append(LargeModule(self.ns))
        self.branch.append(SmallModule(self.ns))
        
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

    def forward(self, x, branch=0, fea_out=False):
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        branch_fea, mask_or_feas = self.branch[branch](z)

        
        z = F.relu(self.tail[0](branch_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        # if fea_out:
        #     return y, mask_or_feas

        return y, mask_or_feas

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
            feas.append(merge_fea)
        
        z = F.relu(self.tail[0](merge_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        # if fea_out:
        #     return y, feas

        return y, feas
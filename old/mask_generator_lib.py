import torch
import torch.nn as nn

class GradientSobelFilter:
    def __init__(self, n_groups=1):
        self.n_groups = n_groups

        self.sfx = nn.Conv2d(self.n_groups, self.n_groups, 3, 1, 1, bias=False, groups=self.n_groups)
        print(self.sfx.weight.size())
        self.sfx.weight.data = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).expand([self.n_groups,1,3,3])
        self.sfx.to('cuda')

        self.sfy = nn.Conv2d(self.n_groups, self.n_groups, 3, 1, 1, bias=False, groups=self.n_groups)
        self.sfy.weight.data = torch.Tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).expand([self.n_groups,1,3,3])
        self.sfy.to('cuda')

    def generate_mask(self, x, p, b=0):
        grad_map_x = self.sfx.forward(x)
        grad_map_y = self.sfy.forward(x)
        grad = abs(grad_map_x) + abs(grad_map_y)

        if b > 0:
            border_mask = torch.zeros_like(grad)
            border_mask[:,:,b:-b,b:-b] = 1.0

            grad = grad * border_mask

        grad_sorted, _ = torch.sort(grad.view(-1), descending=True)
        grad_sorted_index = min(max(int(p * torch.numel(grad)), 0), torch.numel(grad)-1)
        grad_sorted_threshold = grad_sorted[grad_sorted_index]
        
        merge_map = (grad >= grad_sorted_threshold).type(torch.float32)

        return merge_map

class RandomFlatMasker:
    def __init__(self):
        pass

    def generate_mask(self, mask_shape, p):
        merge_map = torch.rand(mask_shape)
        merge_map = (merge_map < p).type(torch.float32)
        # merge_map = merge_map.repeat([branch_fea_0.shape[0],branch_fea_0.shape[1],1,1])

        merge_map = merge_map.to('cuda')
        return merge_map
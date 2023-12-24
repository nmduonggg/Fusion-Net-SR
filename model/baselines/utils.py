import torch
from torch import nn
from calflops import calculate_flops

def residual_stack(z, x, scale=2):
    _,_,h,w = z.size()
    y = torch.repeat_interleave(torch.repeat_interleave(x,scale,dim=2),scale,dim=3)
    for ih in range(scale):
        for iw in range(scale):
            y[:,:,ih:h*scale:scale, iw:w*scale:scale] += z[:,ih*scale+iw:ih*scale+iw+1,:,:]
    return y

def calc_flops(model: nn.Module):
    batch_size = 1
    input_shape = (batch_size, 1, 21, 21)
    flops, macs, params = calculate_flops(model=model, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
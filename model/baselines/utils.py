import torch

def residual_stack(z, x, scale=2):
    _,_,h,w = z.size()
    y = torch.repeat_interleave(torch.repeat_interleave(x,scale,dim=2),scale,dim=3)
    for ih in range(scale):
        for iw in range(scale):
            y[:,:,ih:h*scale:scale, iw:w*scale:scale] += z[:,ih*scale+iw:ih*scale+iw+1,:,:]
    return y
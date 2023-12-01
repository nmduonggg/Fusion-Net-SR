from .edsr import EDSR
from .rcan import RCAN
from .dgnet import DGNet

def config(args):
    arch = args.core.split("-")
    name = arch[0]

    if name=='EDSR':
        return EDSR(scale=args.scale)
    elif name=='RCAN':
        return RCAN(scale=args.scale)
    elif name=='DGNet':
        return DGNet(scale=args.scale)
    else:
        assert(0), 'No configuration found'
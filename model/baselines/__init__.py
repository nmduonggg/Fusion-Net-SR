from .edsr import EDSR
from .rcan import RCAN
from .dgnet import DGNet
from .smsr_v2 import SMSRV2
from .smsr import SMSR
from .original_smsr.smsr import SMSR as OriginalSMSR
from .dgnet_sr import DGNetSR

def config(args):
    arch = args.core.split("-")
    name = arch[0]

    if name=='EDSR':
        return EDSR(scale=args.scale)
    elif name=='RCAN':
        return RCAN(scale=args.scale)
    elif name=='DGNet':
        return DGNet(scale=args.scale)
    elif name=='SMSR':
        return SMSR(scale=args.scale)
    elif name=='SMSRv2':
        return SMSRV2(scale=args.scale)
    elif name=='OriginalSMSR':
        return OriginalSMSR(scale=args.scale)
    elif name=="DGNetSR":
        return DGNetSR(scale=args.scale, tile=args.tile)
    else:
        assert(0), 'No configuration found'
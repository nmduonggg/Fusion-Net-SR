from .SuperNet_4s import SuperNet
from .SuperNet_udl import SuperNet_udl
from .SuperNet_KUL import SuperNet_kul

def config(args):
    arch = args.core.split("-")
    name = arch[0]

    if name=='SuperNet_4s':
        return SuperNet()
    elif name=='SuperNet_udl':
        return SuperNet_udl()
    elif name=='SuperNet_kul':
        return SuperNet_kul()
    else:
        assert(0), 'No configuration found'
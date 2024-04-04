from .SuperNet_4s import SuperNet
from .SuperNet_KUL import SuperNet_kul
from .SuperNet_share import SuperNet_share
from .SuperNet_depend import SuperNet_depend

def config(args):
    arch = args.core.split("-")
    name = arch[0]

    if name=='SuperNet_4s':
        return SuperNet()
    elif name=='SuperNet_udl':
        return SuperNet_udl()
    elif name=='SuperNet_kul':
        return SuperNet_kul()
    elif name=='SuperNet_share':
        return SuperNet_share()
    elif name=='SuperNet_depend':
        return SuperNet_depend()
    else:
        assert(0), 'No configuration found'
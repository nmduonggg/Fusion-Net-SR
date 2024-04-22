from .SuperNet_4s import SuperNet
from .SuperNet_separate import SuperNet_separate
from .SuperNet_share import SuperNet_share
from .SuperNet_depend import SuperNet_depend
from .SuperNet_UDL import SuperNet_udl
from .SuperNet_kul import SuperNet_kul

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
    elif name=='SuperNet_udl':
        return SuperNet_udl()
    elif name=='SuperNet_separate':
        return SuperNet_separate()
    else:
        assert(0), 'No configuration found'
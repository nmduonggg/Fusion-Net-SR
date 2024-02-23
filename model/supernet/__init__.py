from .SuperNet_4s import SuperNet

def config(args):
    arch = args.core.split("-")
    name = arch[0]

    if name=='SuperNet_4s':
        return SuperNet()
    else:
        assert(0), 'No configuration found'
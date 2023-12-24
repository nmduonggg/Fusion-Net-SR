import time

def set_template(args):

    if  args.template == 'EDSR':
        print('[INFO] Template found (EDSR-like SR)')
        args.style='Y'
        args.testset_tag='Set14B'
        args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
        args.rgb_range=1.0
        args.scale=2
        args.core='EDSR'
        args.weight='/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/EDSR/_best.t7'
    elif  args.template == 'RCAN':
        print('[INFO] Template found (RCAN-like SR)')
        args.style='Y'
        args.testset_tag='Set14B'
        args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
        args.rgb_range=1.0
        args.scale=2
        args.core='RCAN'
        args.weight='/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/RCAN/_best.t7'
    elif  args.template == 'DGNet':
        print('[INFO] Template found (DGNet-like SR)')
        args.style='Y'
        args.testset_tag='Set14B'
        args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
        args.rgb_range=1.0
        args.scale=2
        args.core='DGNet'
        args.weight='/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/DGNet/_best.t7'
    elif  args.template == 'SMSR':
        print('[INFO] Template found (SMSR-like SR)')
        args.style='Y'
        args.testset_tag='Set14B'
        args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
        args.rgb_range=1.0
        args.scale=2
        args.core='SMSR'
        args.weight='./checkpoints/SMSR/_best.t7'
    else:
        print('[ERRO] Template not found')
        assert(0)

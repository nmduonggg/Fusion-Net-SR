import time

def set_template(args):

	if  args.template == 'EDSR':
		print('[INFO] Template found (EDSR-like SR)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=128
		args.epoch_step=100
		args.max_epochs=300
		args.loss='L1'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/mnt/disk1/nmduong/FusionNet/data/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='EDSR'
	elif  args.template == 'RCAN':
		print('[INFO] Template found (RCAN-like SR)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=64
		args.epoch_step=100
		args.val_each=2
		args.max_epochs=300
		args.loss='L1'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/mnt/disk1/nmduong/FusionNet/data/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='RCAN'
	elif  args.template == 'DGNet':
		print('[INFO] Template found (DGNet-like SR)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=128
		args.epoch_step=100
		args.val_each=2
		args.max_epochs=300
		args.loss='L1'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/mnt/disk1/nmduong/FusionNet/data/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='DGNet'
	elif  args.template == 'SMSR':
		print('[INFO] Template found (SMSR-like SR)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=256
		args.epoch_step=100
		args.val_each=2
		args.max_epochs=300
		args.loss='L1'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/mnt/disk1/nmduong/FusionNet/data/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='SMSR'
	elif  args.template == 'OriginalSMSR':
		print('[INFO] Template found (Original SMSR SR)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=128
		args.epoch_step=100
		args.val_each=2
		args.max_epochs=300
		args.loss='L1'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/mnt/disk1/nmduong/FusionNet/data/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='OriginalSMSR'

	else:
		print('[ERRO] Template not found')
		assert(0)

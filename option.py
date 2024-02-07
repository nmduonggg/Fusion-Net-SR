import argparse
parser = argparse.ArgumentParser(description="Image Super-Resolution Trainer (clean)", fromfile_prefix_chars="@")

#training hyper-param
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr *= lr_decay_ratio after epoch_steps")
parser.add_argument("--weight_decay",type=float, default=1e-4, help="Weight decay, Default: 1e-4")

parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--epoch_step", type=int, default=20, help="epochs after which lr is decayed")
parser.add_argument("--start_epoch", type=int, default=0, help="starting point")
parser.add_argument("--max_epochs", type=int, default=80, help="total epochs to run")
parser.add_argument("--loss", default="L2", help="loss function")
parser.add_argument("--val-each", type=int, default='5', help='Validation each n epochs')
parser.add_argument("--weight", help='Weight path')

# DGNet hyper-parameters
parser.add_argument("--lbda", type=float, default=5, help='penalty factor of the L2 loss for mask')
parser.add_argument("--gamma", type=float, default=1, help="penalty factor of the L2 loss for balance gate")
parser.add_argument("--den_target", type=float, default=0.5, help="target density of the mask")
parser.add_argument("--tile", type=int, default=1, help="tile size of DGNetSR module, useless for others")

parser.add_argument("--optimizer", default="SGD", help="optimizer")
#--sgd
parser.add_argument("--momentum", type=float, default=0.9, help="learning rate")
#--adam

#data
parser.add_argument("--max_load", default=0, type=int, help="max number of samples to use; useful for reducing loading time during debugging; 0 = load all")
parser.add_argument("--style", default="Y", help="Y-channel or RGB style")
parser.add_argument("--trainset_tag", default="SR291B", help="train data directory")
parser.add_argument("--trainset_patch_size", type=int, default=21, help="train data directory")
parser.add_argument("--trainset_preload", type=int, default=0, help="train data directory")
parser.add_argument("--trainset_dir", default="/home/dataset/sr291_21x21_dn/2x/", help="train data directory")
parser.add_argument("--testset_tag", default="Set14B", help="train data directory")
parser.add_argument("--testset_dir", default="/home/dataset/set14_dnb/2x/", help="test data directory")

#model
parser.add_argument("--rgb_range", type=float, default=1.0, help="int/float images")
parser.add_argument("--scale", type=int, default=2, help="scaling factor")
parser.add_argument("--core", default="SMSR_normal", help="core model (template specified in sr_mask_core.py)")
parser.add_argument("--checkpoint", default=None, help="checkpoint to load core from")

#eval
parser.add_argument("--eval_tag", default="psnr", help="evaluation tag; available: \"psnr, ssim\"")

#output
parser.add_argument("--cv_dir", default="checkpoints", help="checkpoint directory (models and logs are saved here)")

#template
parser.add_argument("--template", default=None)

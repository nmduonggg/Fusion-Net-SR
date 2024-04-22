import os
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
import wandb
import cv2
import matplotlib.pyplot as plt
import numpy as np

#custom modules
import data
import evaluation
import loss
import model.supernet as supernet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils
from option import parser
from template import test_baseline_t as template


args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=0)

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
if args.weight:
    args.weight = os.path.join(args.cv_dir, "SUPERNET_KUL", args.template+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}', '_best_wunc.t7')
    # args.weight='/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/SUPERNET_KUL/SuperNet_kulnblock-1_lbda0.0_gamma0.2_den0.7/_last.t7'
    print(f"[INFO] Load weight from {args.weight}")
    core.load_state_dict(torch.load(args.weight))
core.cuda()

loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.analyze_dir, "SUPERNET_KUL", name+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}', args.testset_tag)
print('Load from: ', out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def test():
    
    core.eval()
    
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()
        
        before_unc, after_unc = core.get_uncertainty_before_after_fusion(x, keep=0.5, T=10)
        print('before:\n', before_unc)
        print('after:\n', after_unc)

if __name__ == '__main__':
    test()
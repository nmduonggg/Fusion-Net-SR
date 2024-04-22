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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)
args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
if args.weight:
    args.weight = os.path.join(args.cv_dir, "SUPERNET_KUL", args.template+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}', '_best_wunc.t7')
    # args.weight='/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/SUPERNET_KUL/SuperNet_kulnblock-1_lbda0.0_gamma0.2_den0.7/_last.t7'
    print(f"[INFO] Load weight from {args.weight}")
    core.load_state_dict(torch.load(args.weight, map_location=device))
core.to(device)

loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.analyze_dir, "SUPERNET_KUL", name+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}', args.testset_tag)
print('Load from: ', out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
        

        
# testing

def calc_relative_increase(perfs):
    x0, x1, x15 = perfs
    return (x15 - x0) / (x1 - x0)

def fuse_fix_idxs(idxs, keeps):
    for keep in keeps:
        perfs = fuse_idxs(idxs, keep)
        perf0, perf1, perf15 = perfs
        print(f"{idxs} - {keep}: {perfs} - {calc_relative_increase(perfs)}")
        
        
        
def fuse_idxs(idxs, keep):
    perfs_val = [0, 0, 0, 0, 0]
    #walk through the test set
    core.eval()
    # core.train()
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.to(device)
        yt = yt.to(device)

        with torch.no_grad():
            out = core(x)
        yfs, masks = out
        fy = core.fuse_2_blocks(x, idxs=idxs, keep=keep)
        
        val_loss = [loss_func(yf, yt).item() for yf in yfs]
        val_loss += [loss_func(fy, yt).item()]
            
        perf_v_layers = [evaluation.calculate(args, yf, yt) for yf in yfs]
        perf_v_layers += [evaluation.calculate(args, fy, yt)]
        
        # print(f"Image {batch_idx}: {perf_v_layers}")

        for i in range(len(perfs_val)):
            perfs_val[i] += perf_v_layers[i]

    perfs_val = [p / len(XYtest) for p in perfs_val]
    id1, id2 = idxs
    return [
        perfs_val[id1], perfs_val[id2], perfs_val[-1]
    ]
    
def test():
    keeps = [0.25, 0.5, 0.75, 0.9]
    all_idxs = [
        [0, 1], [1, 2], [2, 3]
    ]
    for idx in all_idxs:
        fuse_fix_idxs(idx, keeps)

if __name__ == '__main__':
    test()
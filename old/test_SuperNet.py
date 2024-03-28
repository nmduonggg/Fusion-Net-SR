import os
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
import wandb
import cv2
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
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
if args.weight:
    core.load_state_dict(torch.load(args.weight))
    print(f"[INFO] Load weight from {args.weight}")
core.cuda()

loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.analyze_dir, "SUPERNET_FULL", name+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
def get_mask(model, id):
    model.eval() 
    masks = model.masked_s
    for i, m in enumerate(masks):
        m = m.squeeze(0)
        m = m.permute(1, 2, 0)
        m = m.cpu().numpy()*255
        # print(m)
        cv2.imwrite(os.path.join(out_dir, f"{id}_{i}.jpeg"), m)
        
# def get_error_map(yf, yt, id, block_id):
#     error = torch.abs(yt - yf)
#     error = error.squeeze(0)
#     error = error.permute(1,2,0)
#     error = error.cpu().numpy()
    
#     error = (error - error.min()) / (error.max() - error.min())
#     # print(error)
#     error = (error*255).round()
#     new_out_dir = os.path.join(out_dir, "mask_to_GT")
#     if not os.path.exists(new_out_dir):
#         os.makedirs(new_out_dir)
#     cv2.imwrite(os.path.join(new_out_dir, f"E_{id}_{block_id}.jpeg"), error)
    
def get_error_btw_F(yfs, id, t=1e-2):
    error_track = []
    for i in range(len(yfs)):
        if i==len(yfs)-1: continue
        error = torch.abs(yfs[i+1] - yfs[i])
        error = error.squeeze(0).permute(1,2,0)
        error = error.cpu().numpy()
        
        error = (error >= t).astype(int)
        print(f"Enhanced pixel in image {id} - block {i} to {i+1} - threshold {t} is {error.mean()*100:.2f}%")
        error_track.append(error.mean())
        error = (error*255).round()
        
        new_out_dir = os.path.join(out_dir, "mask_to_NEXT")
        if not os.path.exists(new_out_dir):
            os.makedirs(new_out_dir)
        res = cv2.imwrite(os.path.join(new_out_dir, f"img_{id}_{i}_to_{i+1}.jpeg"), error)
        if not res: print(res)
    
    return error_track
    
def get_error_map(yf, yt, id, block_id, mask_block, t=1e-2):
    error = torch.abs(yt - yf)
    error = error.squeeze(0)
    error = error.permute(1,2,0)
    
    error = error.cpu().numpy()
    
    # print(error.mean())
    error = (error >= t).astype(int)
    
    acc = None
    if mask_block is not None:
        mask_block = torch.sigmoid(mask_block).round()
        mask_block = mask_block.cpu().numpy()
        tmp_mask = mask_block.reshape(-1)
        tmp_error = error.reshape(-1)
        
        acc = [1 for i in range(error.shape[0]) if tmp_error[i]==tmp_mask[i]]
        acc = np.mean(np.array(acc))    
        # print(acc)
        
        mask_block = (mask_block*255).round()
        acc *= 100
        
    print(f"Fail pixel in image {id} - block {block_id} - threshold {t} is {error.mean()*100:.2f}%")
    error_m = error.mean()
    error = (error*255)
    new_out_dir = os.path.join(out_dir, "mask_to_GT")
    if not os.path.exists(new_out_dir):
        os.makedirs(new_out_dir)
    cv2.imwrite(os.path.join(new_out_dir, f"E_{id}_{block_id}.jpeg"), error)
    if mask_block is not None:
        cv2.imwrite(os.path.join(new_out_dir, f"Pd_{id}_{block_id}.jpeg"), mask_block)
    
    print(f"Image {id} - block {block_id} - threshold {t} accuracy is {acc}%")
    return error_m

# testing

t = 5e-3

def test():
    perfs_val = [0, 0, 0, 0]
    total_val_loss = 0.0
    error_blocks = [0, 0, 0, 0]
    error_nexts = [0, 0, 0]
    #walk through the test set
    core.eval()
    for m in core.modules():
        if hasattr(m, '_prepare'):
            m._prepare()
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        # print("Running test for image", batch_idx)
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            out = core(x)
        
        yfs, masks = out
        
        val_loss = sum([loss_func(yf, yt).item() for yf in yfs]) / 4
        perf_v_layers = [evaluation.calculate(args, yf, yt) for yf in yfs]
        for i, p in enumerate(perf_v_layers):
            perfs_val[i] = perfs_val[i] + p
        total_val_loss += val_loss
        
        # get_mask(core, batch_idx)
        error_next = get_error_btw_F(yfs, batch_idx, t)
        for i, e in enumerate(error_next):
            error_nexts[i] += e
        
        for block_id in range(len(yfs)):
            mask = masks.squeeze(0)[block_id, ...] if block_id < 3 else None
            error_m = get_error_map(yfs[block_id], yt, batch_idx, block_id, mask, t)
            error_blocks[block_id] += error_m
        
        if hasattr(core, "reset_mask"):
            core.reset_mask()

    perfs_val = [p / len(XYtest) for p in perfs_val]
    error_nexts = [e*100 / len(XYtest) for e in error_nexts]
    error_blocks = [e*100 / len(XYtest) for e in error_blocks]
    perf_f = perfs_val[-1]
    total_val_loss /= len(XYtest)
    
    print("-"*80)
    print("PSNR:", perfs_val) 
    print("Error nexts:", error_nexts)
    print("Error blocks:", error_blocks)
    
    print(f"[TEST] Val Loss {total_val_loss} - Val PSNR {perf_f}")

if __name__ == '__main__':
    test()
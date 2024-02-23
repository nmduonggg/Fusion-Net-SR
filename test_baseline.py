import os
import torch
import torch.utils.data as torchdata
import tqdm
import wandb
import cv2
import numpy as np

#custom modules
import data
import evaluation
import loss
import model.baselines as baseline
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
name = arch[0]
core = baseline.config(args)
if args.weight:
    core.load_state_dict(torch.load(args.weight))
    print(f"[INFO] Load weight from {args.weight}")
core.cuda()

loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.analyze_dir, name+f'_tile{args.tile}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}')
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
        
# def get_error_map(yf, yt, id):
#     error = torch.abs(yt - yf)
#     error = error.squeeze(0)
#     error = error.permute(1,2,0)
#     error = error.cpu().numpy()
    
#     error = (error - error.min()) / (error.max() - error.min())
#     print(error)
#     error = (error*255).round()
#     cv2.imwrite(os.path.join(out_dir, f"E_{id}.jpeg"), error)
    
def get_error_map(yf, yt, id):
    error = torch.abs(yt - yf)
    max_ = torch.amax(error, dim=(2, 3), keepdim=True)
    min_ = torch.amin(error, dim=(2, 3), keepdim=True)
    error = (error - min_) / (max_ - min_)
    
    error = error.squeeze(0)
    error = error.permute(1,2,0)
    
    error = error.cpu().numpy()
    
    print(error.mean())
    error = (error*255).round()
    cv2.imwrite(os.path.join(out_dir, f"E_{id}.jpeg"), error)
# testing
def test():
    perf_fs = []
    total_val_loss = 0.0
    #walk through the test set
    core.eval()
    for m in core.modules():
        if hasattr(m, '_prepare'):
            m._prepare()
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            out = core(x)
        
        density=1
        if type(out) is not list:
            yf = out
        else:
            yf, density = out
        
        val_loss = loss_func(yf, yt).item()
        total_val_loss += val_loss
        perf_f = evaluation.calculate(args, yf, yt)
        perf_fs.append(perf_f.cpu())
        
        get_mask(core, batch_idx)
        get_error_map(yf, yt, batch_idx)
        
        if hasattr(core, "reset_mask"):
            core.reset_mask()

    mean_perf_f = torch.stack(perf_fs, 0).mean()
    total_val_loss /= len(XYtest)
    
    print(f"[TEST] Val Loss {total_val_loss} - Val PSNR {mean_perf_f} - Val Density {density}")

if __name__ == '__main__':
    test()
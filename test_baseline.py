import os
import torch
import torch.utils.data as torchdata
import tqdm
import wandb

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

def test():
    psnrs = []
    ssims = []
    total_val_loss = 0.0
    total_sparsity = 0.0
    valid_density = [0]*4
    #walk through the test set
    core.eval()
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            out = core(x)
        
        if type(out) is not list:
            yf = out
        else:
            yf, sparsity = out
            total_sparsity += sparsity.mean().cpu()
        
        val_loss = loss_func(yf, yt).item()
        total_val_loss += val_loss
        psnr, ssim = evaluation.calculate_all(args, yf, yt)
        psnrs.append(psnr.cpu())
        ssims.append(ssim.cpu())
        
        density = core.density
        print(density)
        for i in range(len(density)):
            valid_density[i] += density[i]
        core.reset_density()
    
    for i in range(len(valid_density)):
        valid_density[i] /= len(XYtest)
        print(f'[SPARSITY] B{i+1}: {1.0-valid_density[i]}')

    mean_psnr = torch.stack(psnrs, 0).mean()
    mean_ssim = torch.stack(ssims, 0).mean()
    total_val_loss /= len(XYtest)
    total_sparsity /= len(XYtest)   # actually density -> sparsity = 1 - total_sparsity
    total_sparsity = 1.0 if total_sparsity==0 else total_sparsity

    log_str = f'[INFO] Val PSNR: {mean_psnr:.3f} - Val SSIM: {mean_ssim:.3f} - Sparsity: {(1.0-total_sparsity):.3f} Val L: {total_val_loss}'
    print(log_str)
    
if __name__=='__main__':
    test()
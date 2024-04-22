"""
This training process is inspired from KULNet and Beyesian MC dropout
"""

import os
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
import wandb

#custom modules
import data
import evaluation
import loss
import model.supernet as supernet
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils
from option import parser
from template import train_baseline as template


args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

print('[INFO] load trainset "%s" from %s' % (args.trainset_tag, args.trainset_dir))
trainset = data.load_trainset(args)
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=0, pin_memory=True)

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
if args.weight:
    core.load_state_dict(torch.load(args.weight))
    print(f"[INFO] Load weight from {args.weight}")
core.cuda()

# initialization
lr = args.lr
batch_size = args.batch_size
epochs = args.max_epochs - args.start_epoch

optimizer = Adam(core.parameters(), lr=lr, weight_decay=args.weight_decay)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.cv_dir, name+f'_nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
def get_error_btw_F(yfs):
    error_track = []
    for i in range(len(yfs)):
        if i>=len(yfs)-1: continue
        high_y = yfs[i+1].contiguous()
        low_y = yfs[i].contiguous()
        error_map = torch.abs(high_y - low_y)
        
        max_ = torch.amax(error_map, dim=1, keepdim=True).to(error_map.device)
        min_ = torch.amin(error_map, dim=1, keepdim=True).to(error_map.device)
        eta = (torch.ones_like(max_-min_)*1e-8).to(error_map.device)
        
        error_map = ((error_map - min_ + eta) / (max_ - min_ + eta)).type(torch.FloatTensor)
        error_track.append(error_map)
    
    return error_track

def rescale_masks(masks):
    new_masks = []
    for m in masks:
        pmin = torch.amin(m, dim=1, keepdim=True)
        m = m - pmin + 1
        m.requires_grad = False
        new_masks.append(m)
    
    return new_masks

def loss_kul(yf, yt, mask):
    mean = yf
    eta = 1e-9
    log_var = 2 * torch.log(mask+eta)
    
    # l1 loss
    l1 = loss_func(yf, yt) 
    # kl loss
    l_kl = -0.5 * (torch.ones_like(mean) + log_var - torch.mul(mean, mean) - torch.mul(mask, mask))
    l_kl = 0.001 * torch.mean(l_kl)
    
    return l1 
    

# training
def train():
    
    # init wandb
    if args.wandb:
        wandb.login(key="60fd0a73c2aefc531fa6a3ad0d689e3a4507f51c")
        wandb.init(
            project='Fusion-Net',
            group=name,
            name=name+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}', 
            entity='nmduonggg',
            config=vars(args)
        )
    
    best_perf = -1e9 # psnr
    best_perf_unc = -1e9
    T = 5
    T_epoch = 0
    T_lambda = 0.05
    
    for epoch in range(epochs):
        track_dict = {}
        
        if epoch % args.val_each == 0:
            perfs_val = [0, 0, 0, 0]
            total_val_loss = 0.0
            total_mask_loss = 0.0
            uncertainty = [0, 0, 0, 0]
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
                
                outs_mean, masks = out
                perf_layers_mean = [evaluation.calculate(args, yf, yt) for yf in outs_mean]
                
                l1_loss = 0.0
                mask_loss = 0.0
                lamb = 0.0 if epoch <= T_epoch else T_lambda
                
                for i, yf in enumerate(outs_mean):
                    # l1_loss = l1_loss + loss_func(yf, yt) + 0.001*masks[i].mean()
                    l1_loss = l1_loss + loss_func(yf, yt)
                    # l1_loss = l1_loss + loss_func(outs_var[i], yt)

                    if lamb > 0:
                        core.train()
                        MC_stds = core.creat_MC_std(x, T)
                        mask_loss = mask_loss + loss_func(masks[i], MC_stds[i].cuda())
                        core.eval()
                    
                val_loss = l1_loss + lamb * mask_loss
                total_val_loss += val_loss.item() if torch.is_tensor(val_loss) else val_loss
                total_mask_loss += mask_loss.item() if torch.is_tensor(mask_loss) else mask_loss
                
                for i, p in enumerate(perf_layers_mean):
                    perfs_val[i] += p
                    uncertainty[i] += masks[i].cpu().detach().mean().item()

            perfs_val = [p / len(XYtest) for p in perfs_val]
            uncertainty = [u / len(XYtest) for u in uncertainty]
            total_val_loss /= len(XYtest)
            total_mask_loss /= len(XYtest)

            for i, p in enumerate(perfs_val):
                track_dict["val_perf_"+str(i)] = p
                track_dict["val_unc_"+str(i)] = uncertainty[i]
                
            track_dict["val_l1_loss"] = total_val_loss
            track_dict["val_mask_loss"] = total_mask_loss

            log_str = f'[INFO] Epoch {epoch} - Val L: {total_val_loss}'
            print(log_str)
            # torch.save(core.state_dict(), os.path.join(out_dir, f'E_%d_P_%.3f.t7' % (epoch, mean_perf_f)))
            
            if lamb > 0.0:
                if perfs_val[-1] > best_perf_unc:
                
                    best_perf_unc = perfs_val[-1]
                    torch.save(core.state_dict(), os.path.join(out_dir, '_best_wunc.t7'))
                    print('[INFO] Save best uncertainty performance model %d with performance %.3f' % (epoch, best_perf_unc))

            else:
                if perfs_val[-1] > best_perf:
                    best_perf = perfs_val[-1]
                    torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                    print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_perf))    
        
        # start training 
        
        total_loss = 0.0
        total_mask_loss = 0.0
        perfs = [0, 0, 0, 0]
        uncertainty = [0, 0, 0, 0]
        
        core.train()
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
            
            # intialize
            x  = x.cuda()
            yt = yt.cuda()
            train_loss = 0.0
            
            # inference
            out = core(x)   # outs, density, mask
            outs_mean, masks = out
            
            perf_layers_mean = [evaluation.calculate(args, yf, yt) for yf in outs_mean]
            
            l1_loss = 0.0
            mask_loss = 0.0
            lamb = 0.0 if epoch <= T_epoch else T_lambda
            for i, yf in enumerate(outs_mean):
                # l1_loss = l1_loss + loss_func(yf, yt) + 0.001*masks[i].mean()
                l1_loss = l1_loss + loss_func(yf, yt)
                # l1_loss = l1_loss + loss_func(outs_var[i], yt)
            
                if lamb > 0.0:
                    MC_stds = core.creat_MC_std(x, T)
                    mask_loss = mask_loss + loss_func(masks[i], MC_stds[i].cuda())
            
            train_loss = l1_loss + lamb * mask_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item() if torch.is_tensor(train_loss) else train_loss
            total_mask_loss += mask_loss.item() if torch.is_tensor(mask_loss) else mask_loss
            for i, p in enumerate(perf_layers_mean):
                perfs[i] = perfs[i] + p
                uncertainty[i] = uncertainty[i] + (masks[i]).detach().cpu().mean()

        total_loss /= len(XYtrain)
        perfs = [p / len(XYtrain) for p in perfs]
        uncertainty = [u / len(XYtrain) for u in uncertainty]
        
        for i, p in enumerate(perfs):
            track_dict["perf_"+str(i)] = p
        track_dict["train_loss"] = total_loss
        
        log_str = '[INFO] E: %d | LOSS: %.3f | Uncertainty: %.3f' % (epoch, total_loss, uncertainty[-1])
        print(log_str)
        
        if args.wandb: 
            wandb.log(track_dict)
            
        # save
        torch.save(core.state_dict(), os.path.join(out_dir, '_last.t7'))
        
if __name__ == '__main__':
    train()
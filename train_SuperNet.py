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
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

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

# initialization
lr = args.lr
batch_size = args.batch_size
epochs = args.max_epochs - args.start_epoch

optimizer = Adam(core.parameters(), lr=lr, weight_decay=args.weight_decay)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.cv_dir, name+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
def get_error_btw_F(yfs):
    error_track = []
    for i in range(len(yfs)):
        if i>=len(yfs)-1: continue
        error_map = torch.abs(yfs[i+1].detach() - yfs[i].detach())
        max_ = torch.amax(error_map, dim=(2, 3), keepdim=True)
        min_ = torch.amin(error_map, dim=(2, 3), keepdim=True)
        error_map = (error_map - min_) / (max_ - min_)  # scale to [0, 1]
        error_track.append(error_map)
    
    return error_track
    

# training
def train():
    
    # init wandb
    wandb.init(
        project='Fusion-Net',
        group='SuperNet',
        name=name+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}', 
        entity='nmduonggg',
        config=vars(args)
    )
    
    best_perf = 0.0 # psnr
    
    for epoch in range(epochs):
        total_loss = 0.0
        perfs = [0, 0, 0, 0]
        train_density = 0.0
        track_dict = {}
        
        core.train()
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
            
            # intialize
            x  = x.cuda()
            yt = yt.cuda()
            train_loss = 0.0
            
            # inference
            out = core(x)   # outs, density, mask
            yfs, masks = out
            
            perf_layers = [evaluation.calculate(args, yf, yt) for yf in yfs]
            
            l1_loss = 0.0
            for yf in yfs:
                l1_loss = l1_loss + loss_func(yf, yt)
            l1_loss = l1_loss * 0.25
            train_loss = train_loss + l1_loss
            
            # dense mask
            error_maps = torch.cat(get_error_btw_F(yfs), dim=1)
            mask_loss = F.l1_loss(masks, error_maps)
            mask_gamma = min((epoch/500), 1) * args.gamma if epoch >= 250 else 0
            train_loss = train_loss + mask_gamma*mask_loss
                        
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()
            for i, p in enumerate(perf_layers):
                perfs[i] = perfs[i] + p

        total_loss /= len(XYtrain)
        train_density /= len(XYtrain)
        perfs = [p / len(XYtrain) for p in perfs]
        
        
        if (epoch+1) % args.val_each == 0:
            perfs_val = [0, 0, 0, 0]
            total_val_loss = 0.0
            total_mask_loss = 0.0
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
                
                yfs, masks = out
                
                val_loss = sum([loss_func(yf, yt).item() for yf in yfs]) / 4

                error_maps = torch.cat(get_error_btw_F(yfs), dim=1)
                mask_loss = F.l1_loss(masks, error_maps)
                    
                perf_v_layers = [evaluation.calculate(args, yf, yt) for yf in yfs]
                for i, p in enumerate(perf_v_layers):
                    perfs_val[i] = perfs_val[i] + p
                total_val_loss += val_loss
                total_mask_loss += mask_loss

            perfs_val = [p / len(XYtest) for p in perfs_val]
            total_val_loss /= len(XYtest)
            total_mask_loss /= len(XYtest)

            for i, p in enumerate(perfs_val):
                track_dict["val_perf_"+str(i)] = p
            track_dict["val_l1_loss"] = total_val_loss
            track_dict["val_mask_loss"] = total_mask_loss

            log_str = f'[INFO] Epoch {epoch} - Val L: {total_val_loss}'
            print(log_str)
            # torch.save(core.state_dict(), os.path.join(out_dir, f'E_%d_P_%.3f.t7' % (epoch, mean_perf_f)))
            
            if perfs_val[-1] > best_perf:
                best_perf = perfs_val[-1]
                torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_perf))    
        
        for i, p in enumerate(perfs):
            track_dict["perf_"+str(i)] = p
        track_dict["train_loss"] = total_loss
        track_dict["train_density"] = train_density
        
        log_str = '[INFO] E: %d | LOSS: %.3f | Density: %.3f' % (epoch, total_loss, train_density)
        print(log_str)
        
        wandb.log(track_dict)
        
if __name__ == '__main__':
    train()
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
import model.baselines as baseline
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
core = baseline.config(args)
if args.weight:
    core.load_state_dict(torch.load(args.weight))
    print(f"[INFO] Load weight from {args.weight}")
core.cuda()

# initialization
lr = args.lr
batch_size = args.batch_size
epochs = args.max_epochs - args.start_epoch

optimizer = Adam(core.parameters(), lr=lr, weight_decay=args.weight_decay)
lr_scheduler = ReduceLROnPlateau(optimizer, patience=5)
loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.cv_dir, name+f'_tile{args.tile}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# training
def train():
    
    best_perf = 0.0 # psnr
    
    for epoch in range(epochs):
        total_loss = 0.0
        perfs = []
        train_density = 0.0
        core.train()
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
            
            # intialize
            x  = x.cuda()
            yt = yt.cuda()
            train_loss = 0.0
            
            # inference
            out = core(x)
            yf, sparsity, spatial_mask = out
            
            perf = evaluation.calculate(args, yf, yt)
            
            l1_loss = loss_func(yf, yt)
            train_loss = train_loss + l1_loss
            
            # density target
            sparsity_loss = sparsity.mean()
            train_density += sparsity_loss.detach().cpu().item()
            lambda_sparsity = min((epoch / 500), 1) * args.lbda
            train_loss = train_loss + sparsity_loss*lambda_sparsity
            
            # dense mask
            error_map = torch.abs(yt - yf)
            max_ = torch.amax(error_map, dim=(2, 3), keepdim=True)
            min_ = torch.amin(error_map, dim=(2, 3), keepdim=True)
            error_map = (error_map - min_) / (max_ - min_)
            
            mask_loss = F.l1_loss(spatial_mask, error_map)
            mask_gamma = min((epoch/50), 1) * args.gamma if epoch >= 5 else 0
            train_loss = train_loss + mask_gamma*mask_loss
                        
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()
            perfs.append(perf.cpu())

        total_loss /= len(XYtrain)
        train_density /= len(XYtrain)
        perf = torch.stack(perfs, 0).mean()
        
        if (epoch+1) % args.val_each == 0:
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

                yf, density, logit_s = out
                
                val_loss = loss_func(yf, yt).item()
                total_val_loss += val_loss
                perf_f = evaluation.calculate(args, yf, yt)
                perf_fs.append(perf_f.cpu())

            mean_perf_f = torch.stack(perf_fs, 0).mean()
            total_val_loss /= len(XYtest)

            log_str = f'[INFO] Epoch {epoch} - Val P: {mean_perf_f:.3f} - Val L: {total_val_loss} - Density: {density.cpu().detach()}'
            print(log_str)
            torch.save(core.state_dict(), os.path.join(out_dir, f'E_%d_P_%.3f.t7' % (epoch, mean_perf_f)))
            
            if mean_perf_f > best_perf:
                best_perf = mean_perf_f
                torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_perf))    
        
        log_str = '[INFO] E: %d | P: %.3f | LOSS: %.3f | Density: %.3f' % (epoch, perf, total_loss, train_density)
        print(log_str)
        
if __name__ == '__main__':
    train()
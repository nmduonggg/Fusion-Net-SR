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
from template import train_baseline as template


args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

print('[INFO] load trainset "%s" from %s' % (args.trainset_tag, args.trainset_dir))
trainset = data.load_trainset(args)
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

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

# initialization
lr = args.lr
batch_size = args.batch_size
epochs = args.max_epochs - args.start_epoch

optimizer = Adam(core.parameters(), lr=lr, weight_decay=args.weight_decay)
lr_scheduler = ReduceLROnPlateau(optimizer, patience=5)
# loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.cv_dir, name+f'_regloss_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
def calc_flops(flops_real, flops_mask, flops_ori, batch_size):
    # total sparsity
    flops_tensor, flops_heads = flops_real[0], flops_real[1]
    # block flops
    flops_conv = flops_tensor[0:batch_size,:].mean(0).sum()
    flops_mask = flops_mask.mean(0).sum()
    flops_ori = flops_ori.mean(0).sum() + flops_heads.mean()
    flops_real = flops_conv + flops_mask + flops_heads.mean() 
    # loss
    return (flops_real.cpu() / flops_ori.cpu()), flops_real.cpu()

# training
def train():
    
    best_perf = 0.0 # psnr
    lbda = args.lbda    # penalty factor of the L2 loss for mask
    gamma = args.gamma   # penalty factor of the L2 loss for balance gate
    den_target = args.den_target # target density of the mask
    track_dict = {}
    
    for epoch in range(epochs):
        total_loss, total_l1, total_rloss, total_bloss, mean_flops = 0.0, 0.0, 0.0, 0.0, 0.0
        perfs = []
        core.train()
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
            x  = x.cuda()
            yt = yt.cuda()
            inputs = {"x": x, "label": yt, "den_target": den_target, "lbda": lbda, "gamma": gamma, "p": 0}
            outputs = core.flop_forward(**inputs)
            train_loss = outputs["closs"].mean() + outputs["rloss"].mean() + outputs["bloss"].mean()
            l1_loss = outputs["closs"].mean().item()
            rloss = outputs["rloss"].mean().item()
            bloss = outputs["bloss"].mean().item()
            
            flops_params = {
                "flops_real": outputs["flops_real"],
                "flops_mask": outputs["flops_mask"],
                "flops_ori": outputs["flops_ori"],
                "batch_size": x.size(0),
            }
            flops_ratio, flops_real = calc_flops(**flops_params)
            mean_flops += flops_ratio
            
            yf = outputs["out"]
            perf = evaluation.calculate(args, yf, yt)
                
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()
            total_l1 += l1_loss
            total_rloss += rloss
            total_bloss += bloss
            
            perfs.append(perf.cpu())

        total_loss /= len(XYtrain)
        total_l1 /= len(XYtrain)
        total_rloss /= len(XYtrain)
        total_bloss /= len(XYtrain)
        mean_flops /= len(XYtrain)
        
        perf = torch.stack(perfs, 0).mean()
        
        track_dict['train_loss'] = total_loss
        track_dict['train_l1'] = total_l1
        track_dict['train_rloss'] = total_rloss
        track_dict['train_bloss'] = total_bloss
        track_dict['train_perf'] = perf
        track_dict['train_flops_ratio'] = mean_flops
        track_dict['train_flops'] = flops_real
        
        if (epoch+1) % args.val_each == 0:
            perf_fs = []
            total_val_loss = 0.0
            val_flops_ratio = 0.0
            val_flops_real = 0.0
            density = [0, 0, 0, 0]
            
            #walk through the test set
            core.eval()
            for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
                x  = x.cuda()
                yt = yt.cuda()
                # print(x.cpu().shape)
                # print(yt.cpu().shape)

                with torch.no_grad():
                    # out = core(x)
                    inputs = {"x": x, "label": yt, "den_target": den_target, "lbda": lbda, "gamma": gamma, "p": 0}
                    outputs = core.flop_forward(**inputs)  
                    
                val_loss = outputs["closs"].mean()             
                total_val_loss += val_loss.item()
                flops_params = {
                    "flops_real": outputs["flops_real"],
                    "flops_mask": outputs["flops_mask"],
                    "flops_ori": outputs["flops_ori"],
                    "batch_size": x.size(0),
                }
                flops_ratio, val_flops_real = calc_flops(**flops_params)
                val_flops_ratio += flops_ratio
                val_flops_real += val_flops_real
                
                density = [density[i] + core.density[i] for i in range(len(density))]
                core.reset_density()
                
                yf = outputs["out"]
                perf_f = evaluation.calculate(args, yf, yt)
                perf_fs.append(perf_f.cpu())

            mean_perf_f = torch.stack(perf_fs, 0).mean()
            total_val_loss /= len(XYtest)
            val_flops_ratio /= len(XYtest)
            val_flops_real /= len(XYtest)
            # lr_scheduler.step(total_val_loss)
            
            track_dict['val_perf'] = mean_perf_f
            track_dict['val_loss'] = total_val_loss
            track_dict['val_flops_ratio'] = val_flops_ratio
            track_dict['val_flops_real'] = val_flops_real
            
            for i, d in enumerate(density):
                track_dict[f'density_L{i}'] = d / len(XYtest)
                print(f'L{i}:', track_dict[f'density_L{i}'])

            log_str = f'[INFO] Epoch {epoch} - Val P: {mean_perf_f:.3f} - Val L: {total_val_loss} - Masked FLOPs ratio: {val_flops_ratio:.1f}'
            print(log_str)
            torch.save(core.state_dict(), os.path.join(out_dir, f'E_%d_P_%.3f.t7' % (epoch, mean_perf_f)))
            
            if mean_perf_f > best_perf:
                best_perf = mean_perf_f
                torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_perf))
        
        log_str = '[INFO] E: %d | P: %.3f | LOSS: %.3f' % (epoch, perf, total_loss)
        print(log_str)
        
if __name__ == '__main__':
    train()
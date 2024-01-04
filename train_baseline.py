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
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32)

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
loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.cv_dir, name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# training
def train():
    
    best_perf = 0.0 # psnr
    
    for epoch in range(epochs):
        total_loss = 0.0
        perfs = []
        core.train()
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
            x  = x.cuda()
            yt = yt.cuda()
            out = core(x)
            if type(out) is not list:
                yf = out
            else:
                yf, sparsity = out  # smsr case
                
            perf = evaluation.calculate(args, yf, yt)
            
            train_loss = loss_func(yf, yt)
            if type(out) is list:
                sparsity_loss = sparsity.mean()
                lambda_sparsity = min((epoch / 50), 1) * 0.05         
                train_loss = train_loss + sparsity_loss*lambda_sparsity
                
                # update tau for gumbel softmax
                tau = max(1 - (epoch - 1) / 500, 0.4)
                for m in core.modules():
                    if hasattr(m, '_update_tau'):
                        m._update_tau(tau)
                
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()
            perfs.append(perf.cpu())

        total_loss /= len(XYtrain)
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
                
                if type(out) is not list:
                    yf = out
                else:
                    yf, sparsity = out
                
                val_loss = loss_func(yf, yt).item()
                total_val_loss += val_loss
                perf_f = evaluation.calculate(args, yf, yt)
                perf_fs.append(perf_f.cpu())

            mean_perf_f = torch.stack(perf_fs, 0).mean()
            total_val_loss /= len(XYtest)

            log_str = f'[INFO] Epoch {epoch} - Val P: {mean_perf_f:.3f} - Val L: {total_val_loss}'
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
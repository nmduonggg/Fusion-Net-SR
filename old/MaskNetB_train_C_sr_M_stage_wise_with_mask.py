import os
import torch
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import tqdm

#custom modules
import data
import evaluation
import loss
import model
import optimizer
import utils
from option import parser
from template import train_sr_masknet_t as template

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser.add_argument("--skip-C", action="store_true", help="skip training C phase")

args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch, optim):
    perfs = []

    total_loss = 0
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
        x  = x.cuda()
        yt = yt.cuda()
        
        yf = core.forward(x)

        perf = evaluation.calculate(args, yf, yt)

        loss_func = loss.create_loss_func(args.loss)

        batch_loss = loss_func(yf, yt)
        optim.zero_grad()
        batch_loss.backward()

        optim.step()

        total_loss += batch_loss
        perfs.append(perf.cpu())

    perf = torch.stack(perfs, 0).mean()

    log_str = '[INFO] E: %d | P: %.3f | LOSS: %.3f' % (epoch, perf, total_loss)
    print(log_str)

def train_kd_stage(epoch, optim, target_stage):
    perfs = []

    total_loss = 0
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
        x  = x.cuda()
        yt = yt.cuda()

        masks = list(range(target_stage))

        fea_teacher, mask_prob = core.forward_stage_wise_sequential_train(x, masks, target_stage)

        gt_prob = (fea_teacher != 0).float()

        loss_func = loss.create_loss_func(args.loss)

        batch_loss = loss_func(gt_prob, mask_prob)

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        total_loss += batch_loss

    log_str = f"[INFO] E: {epoch} | target_stage: {target_stage} | LOSS: {total_loss:.3f}"
    print(log_str)

def test(epoch):
    perf_fs = []
    #walk through the test set
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            yf = core.forward(x)
        
        perf_f = evaluation.calculate(args, yf, yt)
        perf_fs.append(perf_f.cpu())

    mean_perf_f = torch.stack(perf_fs, 0).mean()

    log_str = f'[INFO] TS - CORE - P: {mean_perf_f:.3f}'
    print(log_str)

    torch.save(core.state_dict(), args.cv_dir + '/core_ckpt_E_%d_P_%.3f.t7' % (epoch, mean_perf_f))

def test_mask(epoch):
    perf_fs = []
    masks_stat = {}
    #walk through the test set
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            masks = list(range(core.ns))
            yf, masks_out = core.forward_merge_mask(x, masks, mask_out=True)
        
        for mi, m in zip(masks, masks_out):
            if mi not in masks_stat:
                masks_stat[mi] = 0

            masks_stat[mi] += (1.0 - m.sum() / m.numel())

        perf_f = evaluation.calculate(args, yf, yt)
        perf_fs.append(perf_f.cpu())

    mean_perf_f = torch.stack(perf_fs, 0).mean()

    log_str = f'[INFO] TS - MASK - P: {mean_perf_f:.3f}'
    print(log_str)

    for mi in masks_stat:
        masks_stat[mi] /= len(XYtest)
        print(f"[INFO] TS - MASK - Stage {mi}: {masks_stat[mi]:.3f}")

    torch.save(core.state_dict(), args.cv_dir + '/mask_ckpt_E_%d_P_%.3f.t7' % (epoch, mean_perf_f))

print('[INFO] load trainset "%s" from %s' % (args.trainset_tag, args.trainset_dir))
trainset = data.load_trainset(args)
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

core = model.config(args)
core.cuda()

if not args.skip_C:
#train only the Core
    all_params = []
    all_params += core.core.parameters()
    all_params += core.head.parameters()
    all_params += core.tail.parameters()

    optim_phase_1 = optimizer.create_optimizer(all_params, args)
    lr_scheduler_phase_1 = utils.LrScheduler(optim_phase_1, args.lr, args.lr_decay_ratio, args.epoch_step)

    print('[INFO] train core first')
    for epoch in range(args.start_epoch, args.max_epochs+1):
        lr_scheduler_phase_1.adjust_learning_rate(epoch)

        if epoch % 10 == 0:
            test(epoch)

        train(epoch, optim_phase_1)

#train only the Mask
for target_stage in range(core.ns):
    sub_params = core.mask.stages[target_stage].parameters()

    optim_phase_2 = optimizer.create_optimizer(sub_params, args)
    lr_scheduler_phase_2 = utils.LrScheduler(optim_phase_2, args.lr, args.lr_decay_ratio, args.epoch_step)

    print('[INFO] train mask')
    for epoch in range(args.start_epoch, args.max_epochs+1):
        lr_scheduler_phase_2.adjust_learning_rate(epoch)

        if epoch % 10 == 0:
            test_mask(epoch)

        train_kd_stage(epoch, optim_phase_2, target_stage)
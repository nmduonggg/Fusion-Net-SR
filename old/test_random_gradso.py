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
from template import test_sr_fusionnet_t as template
import numpy as np

from mask_generator_lib import GradientSobelFilter, RandomFlatMasker

args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

def merge_random():
    for psi in range(11):
        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                merge_map = rfm.generate_mask([1, 1, x.shape[2], x.shape[3]], psi/10)
                masks = {i: merge_map for i in range(core.ns)}
                yf = core.forward_merge_mask(x, masks)

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE RANDOM - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)

def merge_gradient():
    for psi in range(11):
        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                merge_map = gsf.generate_mask(x, psi/10)
                masks = {i: merge_map for i in range(core.ns)}
                yf = core.forward_merge_mask(x, masks)

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE GRADSO - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)

def merge_single_layer(li):
    for psi in range(11):
        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                merge_map = gsf.generate_mask(x, psi/10)
                masks = {i: 1.0 for i in range(core.ns)}
                masks[li] = merge_map
                yf = core.forward_merge_mask(x, masks)

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE SINGLE {li} - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)

def merge_diff_p2():
    for psi_0 in range(11):
        for psi_1 in range(11):
            perf_fs = []
            #walk through the test set
            for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
                x  = x.cuda()
                yt = yt.cuda()

                with torch.no_grad():
                    merge_map_0 = gsf.generate_mask(x, psi_0/10)
                    merge_map_1 = gsf.generate_mask(x, psi_1/10)
                    masks = {0: merge_map_0, 1: merge_map_1}
                    yf = core.forward_merge_mask(x, masks)

                perf_f = evaluation.calculate(args, yf, yt)
                perf_fs.append(perf_f.cpu())

            mean_perf_f = torch.stack(perf_fs, 0).mean()

            log_str = f'[INFO] TS - MERGE 2S 0:{psi_0} 1:{psi_1} - P: {mean_perf_f:.3f}'
            print(log_str)

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

core = model.config(args)
core.cuda()

rfm = RandomFlatMasker()
gsf = GradientSobelFilter()

merge_random()
merge_gradient()
merge_single_layer(0)
merge_single_layer(1)
merge_diff_p2()
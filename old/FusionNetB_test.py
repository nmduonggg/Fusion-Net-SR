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

parser.add_argument("--mask-border", type=int, default=0, help="mask border removal")

args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

def test(branches=[]):
    for bri in branches:

        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf = core.forward(x, branch=bri)
            
            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - BRANCH_ID: {bri} - P: {mean_perf_f:.3f}'
        print(log_str)

def merge_gradient():
    for psi in range(11):
        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                merge_map = gsf.generate_mask(x, psi/10, b=args.mask_border)
                masks = {i: merge_map for i in range(core.ns)}
                yf = core.forward_merge_mask(x, masks)

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE GRADSO - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)


def merge_stage_close(mask_gen):
    for psi in range(11):
        perf_fs = []

        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()
            
            yf = core.forward_stage_wise_close_mask(x, mask_gen, psi/10, core.ns) #target_stage = core.ns to run all and return sr image

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE CLOSE - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)


# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

core = model.config(args)
core.cuda()

rfm = RandomFlatMasker()
gsf = GradientSobelFilter()
gsf2 = GradientSobelFilter(n_groups=16)

test(branches=[0,1])
merge_gradient()
merge_stage_close(gsf2)
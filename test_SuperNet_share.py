import os
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
import wandb
import cv2
import matplotlib.pyplot as plt
import numpy as np

#custom modules
import data
import evaluation
import loss
import model.supernet as supernet
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
name = args.template
core = supernet.config(args)
if args.weight:
    args.weight = os.path.join(args.cv_dir, "SUPERNET_SHARE", args.template+f'_nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}', '_best_wunc.t7')
    # args.weight='/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/SUPERNET_KUL/SuperNet_kulnblock-1_lbda0.0_gamma0.2_den0.7/_last.t7'
    print(f"[INFO] Load weight from {args.weight}")
    core.load_state_dict(torch.load(args.weight))
core.cuda()

loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.analyze_dir, "SUPERNET_KUL", name+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}', args.testset_tag)
print('Load from: ', out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
def gray2heatmap(image):
    # colormap = plt.get_cmap('inferno')
    # heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    
    return heatmap
    
def get_mask(model, id):
    model.eval() 
    masks = model.masked_s
    for i, m in enumerate(masks):
        m = m.squeeze(0)
        m = m.permute(1, 2, 0)
        m = m.cpu().numpy()*255
        # print(m)
        cv2.imwrite(os.path.join(out_dir, f"{id}_{i}.jpeg"), m)
        
def process_unc_map(masks, to_heatmap=True, rescale=True, abs=True):
    """
    use for mask with value range [-1, inf]
    apply sigmoid and rescale
    """      
    masks = torch.stack(masks, dim=0)
    if abs: masks = masks.abs()
    
    pmin = torch.min(masks)
    pmax = torch.max(masks)
    agg_mask = 0
    masks_numpy = []
    for i in range(4):
        # mask = torch.abs(mask)
        mask = masks[i, ...]
        
        # print(f'Mask {i}:', torch.mean(mask))
        if rescale:
            mask = (mask - pmin) / (pmax - pmin)
        mask = mask.squeeze(0).permute(1,2,0)
        agg_mask += mask
        mask = mask.cpu().numpy()
        # mask = mask.round()
        if rescale:
            mask = (mask*255).round().astype(np.uint8) 
        if to_heatmap:
            mask = gray2heatmap(mask)  # gray -> rgb
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        masks_numpy.append(mask)
        
    return masks_numpy

def visualize_unc_map(masks, id, val_perfs):
    new_out_dir = out_dir
    os.makedirs(new_out_dir, exist_ok=True)
    save_file = os.path.join(new_out_dir, f"img_{id}_mask.jpeg")
    
    masks_np = process_unc_map(masks, False, False, False)
    masks_np_percentile = [(m > np.percentile(m, 75)) for m in masks_np]
    masks_np_percentile = process_unc_map(masks)
    # mask_np_percentile = (mask[mask_np_percentile[i]] for i, mask in enumerate(masks_np))
    
    fig, axs = plt.subplots(1, len(masks_np_percentile), 
                            tight_layout=True, figsize=(60, 20))
    for i, m in enumerate(masks_np_percentile):
        axs[i].imshow(m)
        axs[i].axis('off')
        axs[i].set_title(f'block {i} - perf {val_perfs[i].detach().item()}')
        
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    
    masks_np = process_unc_map(masks, False, True)
    new_out_dir = os.path.join(out_dir, "Mask_Diff")
    os.makedirs(new_out_dir, exist_ok=True)
    
    save_file = os.path.join(new_out_dir, f"img_{id}_mask_diff.jpeg")
    fig, axs = plt.subplots(1, len(masks_np)-1, 
                            tight_layout=True, figsize=(60, 20))
    for i, m in enumerate(masks_np):
        if i==len(masks_np)-1: continue
        axs[i].imshow((m > masks_np[i+1]).astype(int)*255)
        axs[i].axis('off')
        axs[i].set_title(f'block {i} - perf {val_perfs[i].detach().item()}')
        
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    
    

def visualize_histogram_im(masks, id):
    
    ims = process_unc_map(masks, False, True)
    new_out_dir = out_dir
    os.makedirs(new_out_dir, exist_ok=True)
    save_file = os.path.join(new_out_dir, f"img_{id}_hist.jpeg")

    # calculate mean value from RGB channels and flatten to 1D array
    vals = [im.mean(axis=2).flatten() for im in ims]
    # plot histogram with 255 bins
    fig, axs = plt.subplots(1, len(vals), sharey=True, 
                            tight_layout=True, figsize=(60, 20))
    for i, val in enumerate(vals):
        axs[i].hist(val, 255, edgecolor='black')
        axs[i].set_xlim(0, 255)
        axs[i].set_title(f'block {i} - mean {val.mean()}')
        
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    
def get_error_btw_F(yfs, id):
    error_track = []
    all_errors = []
    
    for i in range(len(yfs)):
        if i==len(yfs)-1: continue
        error = torch.abs(yfs[i+1]-yfs[i])
        all_errors.append(error)
    all_errors = torch.stack(all_errors)
    pmin = torch.min(all_errors)
    pmax = torch.max(all_errors)
    
    fig, axs = plt.subplots(1, len(yfs)-1, figsize=(40, 10))
    for i in range(len(yfs)):
        if i==len(yfs)-1: continue
        error = torch.abs(yfs[i+1] - yfs[i])
        error = error.squeeze(0).permute(1,2,0)
        
        print(f"Difference in image {id} - block {i} to {i+1} is {error.mean():.9f}")
        error_track.append(error.mean())
        
        error = (error - pmin) / (pmax - pmin)
        error = error.cpu().numpy()
        error = (error*255).round()
        
        new_out_dir = os.path.join(out_dir, "mask_to_NEXT")
        if not os.path.exists(new_out_dir):
            os.makedirs(new_out_dir)
        axs[i].imshow(error)
        axs[i].set_title(f"{i}_to_{i+1}")
        
    save_file = os.path.join(new_out_dir, f"img_{id}_error_btw.jpeg")
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()

    return error_track

def visualize_error_enhance(error_wgt, id):
    enhance_map = []
    new_out_dir = os.path.join(out_dir, "Error_Enhanced")
    os.makedirs(new_out_dir, exist_ok=True)
    for i, e in enumerate(error_wgt):
        if i==len(error_wgt)-1: continue
        e = e.cpu().numpy()
        e1 = error_wgt[i+1].cpu().numpy()
        enhance_e = e - e1
        enhance_map.append((enhance_e > np.percentile(enhance_e, 90)).astype(np.uint8)*255)
    
    fig, axs = plt.subplots(1, len(enhance_map), sharey=True, 
                            tight_layout=True, figsize=(60, 20))
    for i, e in enumerate(enhance_map):
        axs[i].imshow(e)
        axs[i].set_title(f'{i}_to{i+1}')
        
    save_file = os.path.join(new_out_dir, f"img_{id}_enhance_error.jpeg")
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    
def visualize_error_map(yfs, yt, id):
    errors = []
    save_file = os.path.join(out_dir, f"img_{id}_error.jpeg")
    for yf in yfs:
        error = torch.abs(yt - yf)
        error = error.squeeze(0)
        error = error.permute(1,2,0)
        errors.append(error)
        
    visualize_error_enhance(errors, id)
        
    pmin = torch.min(torch.stack(errors))
    pmax = torch.max(torch.stack(errors))
    
    ep = []
    for e in errors:
        e = (e - pmin) / (pmax - pmin)
        e = e.cpu().numpy()
        e = (e * 255.).astype(np.uint8)
        ep.append(e)
        
    fig, axs = plt.subplots(1, len(ep), sharey=True, 
                            tight_layout=True, figsize=(60, 20))
    for i, e in enumerate(ep):
        axs[i].imshow(e)
        axs[i].set_title(f'block {i} - error {e.mean()}')
    
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
        
# testing

t = 5e-3

def test():
    perfs_val = [0, 0, 0, 0]
    uncertainty_val = [0, 0, 0, 0]
    total_val_loss = 0.0
    total_mask_loss = 0.0
    #walk through the test set
    core.eval()
    # core.train()
    for m in core.modules():
        if hasattr(m, '_prepare'):
            m._prepare()
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            out = core(x)
        
        yfs, masks = out
        visualize_histogram_im(masks, batch_idx)
        visualize_error_map(yfs, yt, batch_idx)
        get_error_btw_F(yfs, batch_idx)
        
        val_loss = sum([loss_func(yf, yt).item() for yf in yfs]) / 4

        # error_maps = torch.cat(get_error_btw_F(yfs), dim=1).cuda()
        # mask_loss = F.binary_cross_entropy_with_logits(masks, error_maps)
            
        perf_v_layers = [evaluation.calculate(args, yf, yt) for yf in yfs]
        visualize_unc_map(masks, batch_idx, perf_v_layers)
        # print(f'{batch_idx}: ', perf_v_layers)
        for i, p in enumerate(perf_v_layers):
            perfs_val[i] = perfs_val[i] + p
            uncertainty_val[i] = uncertainty_val[i] + torch.exp(masks[i]).contiguous().cpu().mean()
        total_val_loss += val_loss
        # total_mask_loss += mask_loss

    perfs_val = [p / len(XYtest) for p in perfs_val]
    print(perfs_val)
    uncertainty_val = [u / len(XYtest) for u in uncertainty_val]
    total_val_loss /= len(XYtest)
    total_mask_loss /= len(XYtest)

if __name__ == '__main__':
    test()
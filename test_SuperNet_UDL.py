import os
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
import wandb
import cv2
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
    args.weight = os.path.join(args.cv_dir, "SUPERNET_UDL", args.template+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}', '_best.t7')
    core.load_state_dict(torch.load(args.weight))
    print(f"[INFO] Load weight from {args.weight}")
core.cuda()

loss_func = loss.create_loss_func(args.loss)

# working dir
out_dir = os.path.join(args.analyze_dir, "SUPERNET_UDL", name+f'nblock{args.nblocks}_lbda{args.lbda}_gamma{args.gamma}_den{args.den_target}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
def get_mask(model, id):
    model.eval() 
    masks = model.masked_s
    for i, m in enumerate(masks):
        m = m.squeeze(0)
        m = m.permute(1, 2, 0)
        m = m.cpu().numpy()*255
        # print(m)
        cv2.imwrite(os.path.join(out_dir, f"{id}_{i}.jpeg"), m)
        
def gray2redblue(gray):
    assert gray.shape[-1]==1
    img = cv2.merge((gray, gray, gray))
    
    # create 1 pixel red image
    red = np.zeros((1, 1, 3), np.uint8)
    red[:] = (0,0,255)

    # create 1 pixel blue image
    blue = np.zeros((1, 1, 3), np.uint8)
    blue[:] = (255,0,0)

    # append the two images
    lut = np.concatenate((red, blue), axis=0)

    # resize lut to 256 values
    lut = cv2.resize(lut, (1,256), interpolation=cv2.INTER_CUBIC)

    # apply lut
    result = cv2.LUT(img, lut)
    return result
        
def visualize_unc_map(masks, id):
    """
    use for mask with value range [-1, inf]
    apply sigmoid and rescale
    """      
    new_out_dir = os.path.join(out_dir, "mask_ESU")
    os.makedirs(new_out_dir, exist_ok=True)
    
    for i, mask in enumerate(masks):
        mask = torch.sigmoid(mask)
        mask = mask.squeeze(0).permute(1,2,0)
        mask = mask.cpu().numpy()
        mask = (mask*255).round().astype(np.uint8) 
        # mask = gray2redblue(mask)

        res = cv2.imwrite(os.path.join(new_out_dir, f"esumask_i{id}_b{i}.jpeg"), mask)
        if not res: print(res)
        
    return 
    
    
# def get_error_map(yf, yt, id, block_id):
#     error = torch.abs(yt - yf)
#     error = error.squeeze(0)
#     error = error.permute(1,2,0)
#     error = error.cpu().numpy()
    
#     error = (error - error.min()) / (error.max() - error.min())
#     # print(error)
#     error = (error*255).round()
#     new_out_dir = os.path.join(out_dir, "mask_to_GT")
#     if not os.path.exists(new_out_dir):
#         os.makedirs(new_out_dir)
#     cv2.imwrite(os.path.join(new_out_dir, f"E_{id}_{block_id}.jpeg"), error)
    
def get_error_btw_F(yfs, id, t=1e-2):
    error_track = []
    for i in range(len(yfs)):
        if i==len(yfs)-1: continue
        error = torch.abs(yfs[i+1] - yfs[i])
        error = error.squeeze(0).permute(1,2,0)
        error = error.cpu().numpy()
        
        error = (error >= t).astype(int)
        print(f"Enhanced pixel in image {id} - block {i} to {i+1} - threshold {t} is {error.mean()*100:.2f}%")
        error_track.append(error.mean())
        error = (error*255).round()
        
        new_out_dir = os.path.join(out_dir, "mask_to_NEXT")
        if not os.path.exists(new_out_dir):
            os.makedirs(new_out_dir)
        res = cv2.imwrite(os.path.join(new_out_dir, f"img_{id}_{i}_to_{i+1}.jpeg"), error)
        if not res: print(res)
    
    return error_track
    
def get_error_map(yf, yt, id, block_id, mask_block, t=1e-2):
    error = torch.abs(yt - yf)
    error = error.squeeze(0)
    error = error.permute(1,2,0)
    
    error = error.cpu().numpy()
    
    # print(error.mean())
    error = (error >= t).astype(int)
    
    acc = None
    if mask_block is not None:
        mask_block = torch.sigmoid(mask_block).round()
        mask_block = mask_block.cpu().numpy()
        tmp_mask = mask_block.reshape(-1)
        tmp_error = error.reshape(-1)
        
        acc = [1 for i in range(error.shape[0]) if tmp_error[i]==tmp_mask[i]]
        acc = np.mean(np.array(acc))    
        # print(acc)
        
        mask_block = (mask_block*255).round()
        acc *= 100
        
    print(f"Fail pixel in image {id} - block {block_id} - threshold {t} is {error.mean()*100:.2f}%")
    error_m = error.mean()
    error = (error*255)
    new_out_dir = os.path.join(out_dir, "mask_to_GT")
    if not os.path.exists(new_out_dir):
        os.makedirs(new_out_dir)
    cv2.imwrite(os.path.join(new_out_dir, f"E_{id}_{block_id}.jpeg"), error)
    if mask_block is not None:
        cv2.imwrite(os.path.join(new_out_dir, f"Pd_{id}_{block_id}.jpeg"), mask_block)
    
    print(f"Image {id} - block {block_id} - threshold {t} accuracy is {acc}%")
    return error_m

# testing

t = 5e-3

def test():
    perfs_val = [0, 0, 0, 0]
    uncertainty_val = [0, 0, 0, 0]
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
        visualize_unc_map(masks, batch_idx)
        
        val_loss = sum([loss_func(yf, yt).item() for yf in yfs]) / 4

        # error_maps = torch.cat(get_error_btw_F(yfs), dim=1).cuda()
        # mask_loss = F.binary_cross_entropy_with_logits(masks, error_maps)
            
        perf_v_layers = [evaluation.calculate(args, yf, yt) for yf in yfs]
        for i, p in enumerate(perf_v_layers):
            perfs_val[i] = perfs_val[i] + p
            uncertainty_val[i] = uncertainty_val[i] + torch.sigmoid(masks[i]).contiguous().cpu().mean()
        total_val_loss += val_loss
        # total_mask_loss += mask_loss

    perfs_val = [p / len(XYtest) for p in perfs_val]
    uncertainty_val = [u / len(XYtest) for u in uncertainty_val]
    total_val_loss /= len(XYtest)
    total_mask_loss /= len(XYtest)

if __name__ == '__main__':
    test()
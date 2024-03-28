import numpy as np
import torch
import os
import tqdm
from torch.utils.data import Dataset


class SR291_Y_testset(Dataset):
    def __init__(self, root, max_load, lr_patch_size=21, scale=2):
        super(SR291_Y_testset, self).__init__()
        n_sample = 100
        if max_load > 0:
            if n_sample > max_load:
                n_sample = max_load
        self.n_sample = n_sample
        self.root = root
        self.lr_patch_size = lr_patch_size
        self.scale = scale
        
    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        
        im_file_name = self.root + 'im_' + str(idx)
        im = np.reshape(np.fromfile(im_file_name, dtype=np.float32), [1, self.lr_patch_size      , self.lr_patch_size      ])
        gt_file_name = self.root + 'gt_' + str(idx)
        gt = np.reshape(np.fromfile(gt_file_name, dtype=np.float32), [1, self.lr_patch_size*self.scale, self.lr_patch_size*self.scale])
        
        return torch.Tensor(im), torch.Tensor(gt)
    
if __name__=='__main__':
    dts = SR291_Y_testset(
        root='/mnt/disk1/nmduong/FusionNet/data/2x/',
        max_load=10,
    )
    print(dts[0])
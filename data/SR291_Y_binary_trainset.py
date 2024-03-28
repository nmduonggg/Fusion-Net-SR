import numpy as np
import torch
import os
import tqdm
from torch.utils.data import Dataset

# class SR291_Y_binary_trainset(Dataset):
#     def __init__(self, root, max_load, lr_patch_size=21, scale=2):
#         super(SR291_Y_binary_trainset, self).__init__()
#         n_sample = len(os.listdir(root)) // 2
#         if max_load > 0:
#             if n_sample > max_load:
#                 n_sample = max_load
#         im = np.zeros([n_sample, 1, lr_patch_size      , lr_patch_size      ])
#         gt = np.zeros([n_sample, 1, lr_patch_size*scale, lr_patch_size*scale])

#         for i in tqdm.tqdm(range(n_sample)):
#             im_file_name = root + 'im_' + str(i)
#             im[i,:,:,:] = np.reshape(np.fromfile(im_file_name, dtype=np.float32), [1, lr_patch_size      , lr_patch_size      ])
                
#             gt_file_name = root + 'gt_' + str(i)
#             gt[i,:,:,:] = np.reshape(np.fromfile(gt_file_name, dtype=np.float32), [1, lr_patch_size*scale, lr_patch_size*scale])
        
#         self.X = torch.Tensor(im)
#         self.Y = torch.Tensor(gt)

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         return self.X[idx], self.Y[idx]
    
class SR291_Y_binary_trainset(Dataset):
    def __init__(self, root, max_load, lr_patch_size=21, scale=2):
        super(SR291_Y_binary_trainset, self).__init__()
        n_sample = len(os.listdir(root)) // 2
        if max_load > 0:
            if n_sample > max_load:
                n_sample = max_load
        self.n_sample = n_sample
        self.root = root
        self.lr_patch_size = lr_patch_size
        self.scale = scale
        
        # X = np.zeros([n_sample, 1, lr_patch_size      , lr_patch_size      ])
        # Y = np.zeros([n_sample, 1, lr_patch_size*scale, lr_patch_size*scale])

        # for i in tqdm.tqdm(range(n_sample//2)):
        #     im_file_name = root + 'im_' + str(i)
        #     X[i,:,:,:] = np.reshape(np.fromfile(im_file_name, dtype=np.float32), [1, lr_patch_size      , lr_patch_size      ])
                
        #     gt_file_name = root + 'gt_' + str(i)
        #     Y[i,:,:,:] = np.reshape(np.fromfile(gt_file_name, dtype=np.float32), [1, lr_patch_size*scale, lr_patch_size*scale])

        # self.X = torch.Tensor(X).type(torch.FloatTensor)
        # self.Y = torch.Tensor(Y).type(torch.FloatTensor)
        
    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        
        # if idx < self.n_sample//2:
        #     return self.X[idx], self.Y[idx]
        
        im_file_name = self.root + 'im_' + str(idx)
        im = np.reshape(np.fromfile(im_file_name, dtype=np.float32), [1, self.lr_patch_size      , self.lr_patch_size      ])
        gt_file_name = self.root + 'gt_' + str(idx)
        gt = np.reshape(np.fromfile(gt_file_name, dtype=np.float32), [1, self.lr_patch_size*self.scale, self.lr_patch_size*self.scale])
        
        return torch.Tensor(im), torch.Tensor(gt)
    
if __name__=='__main__':
    dts = SR291_Y_binary_trainset(
        root='/mnt/disk1/nmduong/FusionNet/data/2x/',
        max_load=10,
    )
    print(dts[0])
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

# class SetN_Y_binary_testset(Dataset):
#     def __init__(self, root, N, scale=2):
#         super(SetN_Y_binary_testset, self).__init__()
#         self.X, self.Y = [], []
#         for i in tqdm.tqdm(range(N)):
#             im_file_name = root + 'im_' + str(i)
#             data = np.fromfile(im_file_name, dtype=np.float32)
#             imw = data[0].astype(np.int32)
#             imh = data[1].astype(np.int32)
#             im_data = np.reshape(data[2:], [1, imh, imw])
#             self.X.append(torch.Tensor(im_data))

#             gt_file_name = root + 'gt_' + str(i)
#             data = np.fromfile(gt_file_name, dtype=np.float32)
#             gtw = data[0].astype(np.int32)
#             gth = data[1].astype(np.int32)
#             gt_data = np.reshape(data[2:], [1, gth, gtw])
#             self.Y.append(torch.Tensor(gt_data))
#             self.N = N

#     def __len__(self):
#         return self.N

#     def __getitem__(self, idx):
#         return self.X[idx], self.Y[idx]
    
class SetN_Y_binary_testset(Dataset):
    def __init__(self, root, N, scale=2):
        super(SetN_Y_binary_testset, self).__init__()
        self.X, self.Y = [], []
        self.root = root
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        try:
            im_file_name = self.root + f'im_{idx}.npy'
            data = np.load(im_file_name)
        except: 
            im_file_name = self.root + f'im_{idx}'
            data = np.fromfile(im_file_name, dtype=np.float32)
        
        data = data.astype(np.float32)
        imw = data[0].astype(np.int32)
        imh = data[1].astype(np.int32)
        
        im_data = np.reshape(data[2:], [1, imh, imw])
        x = im_data
        
        try:
            gt_file_name = self.root + f'gt_{idx}.npy'
            gt = np.load(gt_file_name)
        except:
            gt_file_name = self.root + f'gt_{idx}'
            gt = np.fromfile(gt_file_name, dtype=np.float32)
            
        gt = gt.astype(np.float32)
        gtw = gt[0].astype(np.int32)
        gth = gt[1].astype(np.int32)
        gt_data = np.reshape(gt[2:], [1, gth, gtw])
        y = gt_data
        
        return x, y
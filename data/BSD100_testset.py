import numpy as np
import torch
import tqdm
import skimage.color as sc
import imageio
from torch.utils.data import Dataset

class BSD100_Y_binary_testset(Dataset):
    def __init__(self, root, N, scale=2, style='Y', rgb_range=1.0):
        super(BSD100_Y_binary_testset, self).__init__()
        self.scale = scale
        self.root = root
        self.rgb_range = rgb_range
        self.style = style
        self.N = N

    def __len__(self):
        return self.N
    
    def get_Y_from_RGB(self, rgb_img_file):
        img = imageio.imread(rgb_img_file)
        # stype = Y
        if img.ndim==2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2]==3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(img).float()
        img = img.mul_(self.rgb_range / 255.)
        return img

    def __getitem__(self, idx):
        im_file_name = self.root + 'img_' + str(idx+1).zfill(3) + f'_SRF_{self.scale}_LR.png'
        im_data = self.get_Y_from_RGB(im_file_name)
        x = im_data
        
        gt_file_name = self.root + 'img_' + str(idx+1).zfill(3) + f'_SRF_{self.scale}_HR.png'
        gt_data = self.get_Y_from_RGB(gt_file_name)
        y = gt_data
        
        return x, y
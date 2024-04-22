import numpy as np
import torch

#this is how to deal with "Too many open files" error. Shitty
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import skimage.color as sc
import imageio

from data.DIV2K_testset import DIV2K_testset
from data.DIV2K_trainset import DIV2K_trainset
from data.DIV2K_validset import DIV2K_validset
from data.SetN_testset import SetN_testset
from data.SetN_Y_binary_testset import SetN_Y_binary_testset
from data.SR291_trainset import SR291_trainset
from data.SR291_Y_binary_trainset import SR291_Y_binary_trainset
from data.Set14_RGB_testset import Set14_RGB_testset
from data.BSD100_testset import BSD100_Y_binary_testset
from data.SR291_Y_testset import SR291_Y_testset

def load_trainset(args):
    tag = args.trainset_tag
    if tag == 'SR291B' and args.style == 'Y':
        return SR291_Y_binary_trainset(args.trainset_dir, max_load=args.max_load, lr_patch_size=args.trainset_patch_size, scale=args.scale)
    if tag == 'SR291':
        return SR291_trainset(args.trainset_dir, max_load=args.max_load, lr_patch_size=args.trainset_patch_size, scale=args.scale, style=args.style, rgb_range=args.rgb_range)
    if tag == 'DIV2K':
        return DIV2K_trainset(args.trainset_dir, max_load=args.max_load, lr_patch_size=args.trainset_patch_size, scale=args.scale, style=args.style, preload=args.trainset_preload, rgb_range=args.rgb_range)
    else:
        print('[ERRO] unknown tag and/or style for trainset')
        assert(0)

def load_testset(args):
    tag = args.testset_tag
    if tag == 'Set5B' and args.style == 'Y':
        print('[WARN] RGB range (<rgb_range>) set to 1.0')
        batch_size_test = 1
        return SetN_Y_binary_testset(args.testset_dir, args.N, scale=args.scale), batch_size_test
    elif tag == 'Set14B' and args.style == 'Y':
        print('[WARN] RGB range (<rgb_range>) set to 1.0')
        batch_size_test = 1
        return SetN_Y_binary_testset(args.testset_dir, args.N, scale=args.scale), batch_size_test
    elif tag== 'BSD100' and args.style=='Y':
        print('[WARN] RGB range (<rgb_range>) set to 1.0')
        batch_size_test = 1
        return BSD100_Y_binary_testset(args.testset_dir, args.N, scale=args.scale), batch_size_test
    elif tag=='Set14RGB' and args.style=='RGB':
        print('[WARN] RGB range (<rgb_range>) set to 1.0')
        batch_size_test = 1
        return Set14_RGB_testset(root=args.testset_dir, scale=args.scale, style=args.style, rgb_range=args.rgb_range), batch_size_test        
    elif tag == 'SetN':
        batch_size_test = 1
        return SetN_testset(args.testset_dir, scale=args.scale, style=args.style, rgb_range=args.rgb_range), batch_size_test
    elif tag == 'DIV2K-test':
        batch_size_test = 1
        return DIV2K_testset(args.testset_dir, scale=args.scale, style=args.style, rgb_range=args.rgb_range), batch_size_test
    elif tag == 'DIV2K-valid':
        batch_size_test = 1
        return DIV2K_validset(args.testset_dir, scale=args.scale, style=args.style, rgb_range=args.rgb_range), batch_size_test
    elif tag == 'SR291-test':
        batch_size_test=1
        return SR291_Y_testset(args.testset_dir, max_load=args.max_load), batch_size_test
    else:
        print('[ERRO] unknown tag and/or style for testset')
        assert(0)

print('[ OK ] Module "data"')
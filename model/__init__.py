# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from model.FusionNet_7_1s import FusionNet_7_1s
# from model.FusionNet_7_2s import FusionNet_7_2s
# from model.FusionNet_7_3s import FusionNet_7_3s
# from model.FusionNet_7_4s import FusionNet_7_4s
# from model.FusionNetB_7_1s import FusionNetB_7_1s
# from model.FusionNetB_7_2s import FusionNetB_7_2s
# from model.FusionNetB_7_3s import FusionNetB_7_3s
# from model.FusionNetB_7_4s import FusionNetB_7_4s
# from model.FusionNetB_8_1s import FusionNetB_8_1s
# from model.FusionNetB_8_2s import FusionNetB_8_2s
# from model.FusionNetB_8_3s import FusionNetB_8_3s
# from model.FusionNetB_8_4s import FusionNetB_8_4s
# from model.MaskNetB_7_4s import MaskNetB_7_4s
# from model.NewNet_4s import NewNet_4s

# def config(args):
#     arch = args.core.split("-")
#     name = arch[0]

#     if   (name == "FusionNet_7_1s"):
#         core = FusionNet_7_1s(scale=args.scale)
#     elif (name == "FusionNet_7_2s"):
#         core = FusionNet_7_2s(scale=args.scale)
#     elif (name == "FusionNet_7_3s"):
#         core = FusionNet_7_3s(scale=args.scale)
#     elif (name == "FusionNet_7_4s"):
#         core = FusionNet_7_4s(scale=args.scale)
#     elif (name == "FusionNetB_7_1s"):
#         core = FusionNetB_7_1s(scale=args.scale)
#     elif (name == "FusionNetB_7_2s"):
#         core = FusionNetB_7_2s(scale=args.scale)
#     elif (name == "FusionNetB_7_3s"):
#         core = FusionNetB_7_3s(scale=args.scale)
#     elif (name == "FusionNetB_7_4s"):
#         core = FusionNetB_7_4s(scale=args.scale)
#     elif (name == "FusionNetB_8_1s"):
#         core = FusionNetB_8_1s(scale=args.scale)
#     elif (name == "FusionNetB_8_2s"):
#         core = FusionNetB_8_2s(scale=args.scale)
#     elif (name == "FusionNetB_8_3s"):
#         core = FusionNetB_8_3s(scale=args.scale)
#     elif (name == "FusionNetB_8_4s"):
#         core = FusionNetB_8_4s(scale=args.scale)
#     elif (name == "MaskNetB_7_4s"):
#         core = MaskNetB_7_4s(scale=args.scale)
#     elif (name == "NewNet_4s"):
#         core = NewNet_4s(scale=args.scale)
#     else:
#         assert 0, f"[ERRO] unknown model tag {name}"

#     if args.checkpoint is not None:
#         print("[INFO] load core from torch checkpoint: " + args.checkpoint)
#         checkpoint_data = torch.load(args.checkpoint)
#         core.load_state_dict(checkpoint_data, strict=False)

#     return core

# print("[ OK ] Module \"model\"")
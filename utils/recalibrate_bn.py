import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from tqdm import tqdm
from train import RANK

import val as validate  # for end-of-epoch mAP

from utils.dataloaders import create_dataloader
from utils.general import init_seeds, check_dataset, check_img_size, colorstr, check_amp
from utils.torch_utils import torch_distributed_zero_first

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class AdaptiveBNEval(object):
    def __init__(self, model, opt) -> None:
        super().__init__()
        self.model = model
        self.opt = opt
        self.device = opt.device
        self.hyp = opt.hyp
        self.amp = check_amp(model)  # check AMP

        batch_size = opt.batch_size
        cuda = opt.device.type != 'cpu'
        init_seeds(1 + RANK, deterministic=True)
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = check_dataset(opt.data)  # check
        train_path, val_path = data_dict['train'], data_dict['val']
        self.data_dict = data_dict

        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    
        train_loader, _ = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              opt.single_cls,
                                              hyp=opt.hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rank=LOCAL_RANK,
                                              workers=opt.workers,
                                              prefix=colorstr('train: '),
                                              shuffle=True)

        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       opt.single_cls,
                                       hyp=opt.hyp,
                                       cache=opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=opt.workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        model.half().float()  # pre-reduce anchor precision

        self.batch_size = batch_size
        self.imgsz = imgsz
        self.val_loader = val_loader
        self.train_loader = train_loader
        # self.train_loader = val_loader

    def __call__(self, compact_model):
        compact_model.train()
        with torch.no_grad():
            for i, (imgs, targets, paths, _) in tqdm(enumerate(self.train_loader),total=len(self.train_loader)):
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    _ = compact_model(imgs)  # forward
                
                if i > 50:
                    break

        results, maps, _ = validate.run(self.data_dict,
                                        batch_size=self.batch_size // WORLD_SIZE * 2,
                                        imgsz=self.imgsz,
                                        half=self.amp,
                                        model=compact_model,
                                        single_cls=self.opt.single_cls,
                                        dataloader=self.val_loader,
                                        plots=False)

        return results[2]

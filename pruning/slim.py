import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate
from models.yolo import *
from models.common import *
from utils.general import set_logging, check_file, intersect_dicts, check_yaml
from utils.torch_utils import select_device, de_parallel
from utils.prune_utils import *
from utils.recalibrate_bn import *

def prune_and_export(model, ignore_idx, opt):
    origin_flops = model_flops(model)
    ignore_conv_idx = [i.replace('bn','conv') for i in ignore_idx]
    maskbndict = {}
    maskconvdict = {}
    with open(opt.cfg, encoding='ascii', errors='ignore') as f:
        oriyaml = yaml.safe_load(f)  # model dict

    pruned_yaml = deepcopy(oriyaml)

    # obtain mask
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if name in ignore_idx:
                mask = torch.ones(module.weight.data.size(0))
            else:
                mask = obtain_filtermask_bn(module, opt.thresh)
            maskbndict[name] = mask
            maskconvdict[name[:-2] + 'conv'] = mask

    pruned_yaml = update_yaml(pruned_yaml, model, ignore_conv_idx, maskconvdict, opt)

    compact_model = Model(pruned_yaml).to(opt.device)
    current_flops = model_flops(compact_model)
    print('slimmed model flops is {}'.format(current_flops))
    weights_inheritance(model, compact_model, from_to_map, maskbndict)

    with open(opt.saved_cfg, "w", encoding='utf-8') as f:
        yaml.safe_dump(pruned_yaml, f, encoding='utf-8', allow_unicode=True, default_flow_style=True, sort_keys=False)
    ckpt = {'epoch': -1,
            'best_fitness': None,
            'model': deepcopy(de_parallel(compact_model)).half(),
            'ema': None,
            'updates': None,
            'optimizer': None,
            'wandb_id': None}
    
    torch.save(ckpt, opt.weights[:-3]+'_slim.pt')
    opt.weights = opt.weights[:-3]+'_slim.pt'
    validate.run(data=opt.data,
                 weights=opt.weights,
                 batch_size=opt.batch_size,
                 imgsz=opt.imgsz,
                 conf_thres=opt.conf_thres,  # confidence threshold
                 iou_thres=opt.iou_thres,
                 workers=opt.workers,
                 plots=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="pretrained_weights/yolov5l.pt", help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5l_slim.yaml', help='model.yaml')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--saved_cfg', type=str, default='models/yolov5l_slimmed.yaml', help='the path to save pruned yaml')
    parser.add_argument('--thresh', type=float, default=0.5, help='bn scaling factor thresh')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_yaml(opt.data)  # check YAML
    data = check_dataset(opt.data)  # check
    set_logging()
    opt.device = select_device(opt.device)

    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    opt.hyp = hyp.copy()

    # Create model
    ckpt = torch.load(opt.weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(opt.cfg or ckpt['model'].yaml, ch=3, anchors=hyp.get('anchors')).to(opt.device)  # create
    exclude = []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=True)  # load
    model.nc = data['nc']

    # Parse Module
    ignore_idx, from_to_map, _ = parse_module_defs(model.yaml)
    prune_and_export(model, ignore_idx, opt)
import argparse
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

from models.yolo import *
from models.common import *
from utils.general import set_logging, check_file, intersect_dicts
from utils.torch_utils import select_device, de_parallel
from utils.prune_utils import *
from utils.recalibrate_bn import *


def rand_prune_and_eval(model, ignore_idx, ratio_num_sum, opt):
    origin_flops = model_flops(model)
    ignore_conv_idx = [i.replace('bn','conv') for i in ignore_idx]
    max_remain_ratio = 1.0
    candidates = 0
    max_mAP = 0
    maskbndict = {}
    maskconvdict = {}
    with open(opt.cfg, encoding='ascii', errors='ignore') as f:
        oriyaml = yaml.safe_load(f)  # model dict
    
    ABE = AdaptiveBNEval(model, opt)

    while True:
        pruned_yaml = deepcopy(oriyaml)
        remain_ratio_list = (np.random.rand(ratio_num_sum) * (1.0 - opt.min_remain_ratio) + opt.min_remain_ratio).tolist()
        idx = 0
        # obtain mask
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name in ignore_conv_idx:
                    mask = torch.ones(module.weight.data.size(0)) # [N, C, H, W]
                else:
                    # rand_remain_ratio = (max_remain_ratio - opt.min_remain_ratio) * (np.random.rand(1)) + opt.min_remain_ratio
                    rand_remain_ratio = remain_ratio_list[idx]
                    idx += 1
                    # rand_remain_ratio = 1.0
                    mask = obtain_filtermask_l1(module, rand_remain_ratio)
                maskbndict[(name[:-4] + 'bn')] = mask
                maskconvdict[name] = mask

        pruned_yaml = update_yaml(pruned_yaml, model, ignore_conv_idx, maskconvdict, opt)

        compact_model = Model(pruned_yaml).to(opt.device)
        current_flops = model_flops(compact_model)
        if (current_flops/origin_flops > opt.remain_ratio+opt.delta) or (current_flops/origin_flops < opt.remain_ratio-opt.delta):
            del compact_model
            del pruned_yaml
            continue
        weights_inheritance(model, compact_model, from_to_map, maskbndict)
        mAP = ABE(compact_model)
        print('mAP@0.5 of candidate sub-network is {:f}'.format(mAP))

        if mAP > max_mAP:
            max_mAP = mAP
            with open(opt.saved_cfg, "w", encoding='utf-8') as f:
                yaml.safe_dump(pruned_yaml, f, encoding='utf-8', allow_unicode=True, default_flow_style=True, sort_keys=False)
                # yaml.dump(pruned_yaml, f, Dumper=ruamel.yaml.RoundTripDumper)
            ckpt = {'epoch': -1,
                    'best_fitness': [max_mAP],
                    'model': deepcopy(de_parallel(compact_model)).half(),
                    'ema': None,
                    'updates': None,
                    'optimizer': None,
                    'wandb_id': None}
            torch.save(ckpt, opt.weights[:-3]+'_ee.pt')

        candidates = candidates + 1
        del compact_model
        del pruned_yaml
        if candidates > opt.max_iter:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="pretrained_weights/yolov5l.pt", help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5l_slim.yaml', help='model.yaml')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--saved_cfg', type=str, default='models/yolov5l_slimmed.yaml', help='the path to save pruned yaml')
    parser.add_argument('--min_remain_ratio', type=float, default=0.25)
    parser.add_argument('--max_iter', type=int, default=1000, help='maximum number of arch search')
    parser.add_argument('--remain_ratio', type=float, default=0.5, help='FLOPs remain ratio')
    parser.add_argument('--delta', type=float, default=0.05, help='flops delta')
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
    ignore_idx, from_to_map, ratio_num_sum = parse_module_defs(model.yaml)
    rand_prune_and_eval(model, ignore_idx, ratio_num_sum, opt)